"""
This script defines the class that can be used for Actionable Feature Tweaking.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
from networkx.drawing.nx_pydot import graphviz_layout
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import networkx as nx
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from core.logmanager import get_logger

TREE_LEAF = -1

class ActionableFeatureTweaker(object):
    """Based on Interpretable Predictions of Tree-based Ensembles via Actionable Feature Tweaking, KDD'17
    Currently supports Tree classifiers built using scikit-learn
    Future plan includes supporting CATBoost and XGB
    Version: 0.0.1
    """

    def __init__(self,
                 model = None,
                 feature_names = None,
                 class_names = None,
                 target_class = None
                 ):
        self.model = model
        self.logger = get_logger("ActionableFeatureTweaker")
        self.feature_names = feature_names
        self.class_names = class_names
        self.target_class = target_class
        self.precision = 3

        # Model is an instance of
        if not (isinstance(self.model, DecisionTreeClassifier) or
                isinstance(self.model, GradientBoostingClassifier) or
                isinstance(self.model, RandomForestClassifier)):
            self.logger.exception("Model must be one of DecisionTreeClassifier, GradientBoostingClassifier, or RandomForestClassifier")
            raise TypeError(
                "Model must be one of DecisionTreeClassifier, GradientBoostingClassifier, or RandomForestClassifier"
            )

        # Must have features that match with the model
        if self.feature_names is not None:
            if len(self.feature_names) != self.model.n_features_in_:
                self.logger.exception(
                    "Length of feature_names, %d does not match number of features, %d"
                    % (len(self.feature_names), self.model.n_features_in_))
                raise ValueError(
                    "Length of feature_names, %d does not match number of features, %d"
                    % (len(self.feature_names), self.model.n_features_in_)
                )
        else:
            self.logger.exception(
                "Feature names must be provided!")
            raise ValueError(
                "Feature names must be provided!"
            )

        # Must have classes that match with the model
        if self.class_names is not None:
            if len(self.class_names) != self.model.n_classes_:
                self.logger.exception(
                    "Length of classes, %d does not match number of classes, %d"
                    % (len(self.class_names), self.model.n_classes_))
                raise ValueError(
                    "Length of classes, %d does not match number of classes, %d"
                    % (len(self.class_names), self.model.n_classes_)
                )
        else:
            self.logger.exception(
                "Feature names must be provided!")
            raise ValueError(
                "Feature names must be provided!"
        )

        # Must have a target class
        if self.target_class is None:
            self.logger.exception(
                "A Target class name must be provided!")
            raise ValueError(
                "A Target class name must be provided!"
            )

        # Get the trees
        self._forest_from_model()

        # get all positive paths for the target class
        self.paths = self._get_positive_paths()

    def _forest_from_model(self):
        self.forest = []
        if isinstance(self.model, DecisionTreeClassifier):
            self.n_forest = 1
            decision_tree = self.model
            tree = self._tree_from_decision_tree(decision_tree)
            self.forest.append(tree)
        else:
            self.n_forest = self.model.n_estimators
            for index in range(0, self.model.n_estimators):
                decision_tree = self.model.estimators_[index]
                # added for debugging
                # self._draw_tree(decision_tree, index)
                tree = self._tree_from_decision_tree(decision_tree)
                self.forest.append({'tree': tree, 'tree_id': index})

    def _tree_from_decision_tree(self, decision_tree):
        G = nx.DiGraph()
        self._tree_recurse(decision_tree.tree_, G, 0, criterion=decision_tree.criterion)
        if not nx.is_tree(G):
            self.logger.exception(
                "Expected a Tree but received a graph!")
            raise ValueError(
                "Expected a Tree but received a graph!"
            )
        return G

    def _tree_recurse(self, tree, G, node_id, criterion, parent=None, depth=0):
        if node_id == TREE_LEAF:
            self.logger.exception(
                "Invalid node_id %s" % TREE_LEAF)
            raise ValueError("Invalid node_id %s" % TREE_LEAF)

        # Add current node to G with properties
        value = self._get_node_value(tree, node_id)
        decision_class = self.class_names[np.argmax(value)]
        samples = tree.n_node_samples[node_id]
        impurity = round(tree.impurity[node_id], self.precision)
        [feature, threshold] = self._get_node_decision_criteria(tree, node_id)
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        G.add_node(node_id,
                   id = node_id,
                   value = value,
                   decision_class = decision_class,
                   samples = samples,
                   threshold = threshold,
                   feature = feature,
                   impurity = impurity,
                   depth = depth,
                   criterion = criterion,
                   left_child = left_child,
                   right_child = right_child)

        # If parent is not null then add edge from parent to me
        if parent is not None:
            G.add_edge(parent,node_id)

        # Recurse on left and right children unless we are on a leaf node
        # scikit-leaf noode has child id -1
        if left_child != TREE_LEAF:
            self._tree_recurse(
                tree,
                G,
                left_child,
                criterion=criterion,
                parent=node_id,
                depth=depth + 1,
            )
            self._tree_recurse(
                tree,
                G,
                right_child,
                criterion=criterion,
                parent=node_id,
                depth=depth + 1,
            )

    def _get_node_value(self, tree, node_id):
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]

        return value

    def _get_node_decision_criteria(self, tree, node_id):
        if tree.children_left[node_id] != TREE_LEAF:
            feature = self.feature_names[tree.feature[node_id]]
            threshold = round(tree.threshold[node_id], self.precision)
        else:
            # Nothong for a leaf node
            feature = None
            threshold = None
        return [feature, threshold]

    def _get_positive_paths(self):
        positive_paths = [None] * self.n_forest
        for f in self.forest:
            tree = f['tree']
            tree_id = f['tree_id']
            paths = self._get_positive_path_on_dt(tree)
            positive_paths[tree_id] = {'tree_id': tree_id, 'tree': tree, 'paths': paths}

        return positive_paths

    def _get_positive_path_on_dt(self, tree):
        """
        A positive path is a path that leads to a leaf node with
        target class. Nodes on the path keeps feature, threshold, and direction.
        To fully characterize the path we consider the following convention
        in path representation. We decide to encode paths as triples:
        (f1, <=, -0.7171), (f2, >, 457.0), (f3, <=, 54.609), (f4, >, -0.059), ...
        With this encoding scheme we aim to represent the following:
        - feature f1 needs to be less than or equal to -0.7171 (feature threshold is negative, direction is <=);
        - feature f2 needs to be greater than 457.0 (feature threshold is positive, direction is >);
        - feature f3 needs to be less than or equal to 54.609 (feature threshold is positive, direction is <=);
        - feature f4 needs to greater than -0.059 (feature threshold is negative, direction is >);
        Direction is "<=" for left child or root and ">" for the right child.
        NOTE: this tree is a nx graph and not a scikit Tree class
        """
        list_of_paths = []
        current_path = []

        node = tree.nodes[0]
        self._recursive_depth_first(tree, node, current_path, list_of_paths)

        return list_of_paths

    def _recursive_depth_first(self, tree, node, current_path, list_of_paths):
        if node['left_child'] == TREE_LEAF:
            if node['decision_class'] == self.target_class:
                list_of_paths.append(current_path)
            return

        # It's not a leaf. Continue recursion
        me = node['id']
        threshold = node['threshold']
        feature = node['feature']

        # First go left
        left_child_id = node['left_child']
        left_child = tree.nodes[left_child_id]
        direction = "<="

        # Append to current path
        current_node = {'id': me,
                        'threshold': threshold,
                        'feature': feature,
                        'direction': direction}

        # Jump - NOTE: [] creates a new instance of list after we return from leaf
        # append on current_path will not work
        self._recursive_depth_first(tree,
                                    left_child,
                                    current_path + [current_node],
                                    list_of_paths
                                    )

        # Then go right
        right_child_id = node['right_child']
        right_child = tree.nodes[right_child_id]
        direction = ">"

        # Append to current path
        current_node = {'id': me,
                        'threshold': threshold,
                        'feature': feature,
                        'direction': direction}

        # Jump
        self._recursive_depth_first(tree,
                                    right_child,
                                    current_path + [current_node],
                                    list_of_paths
                                    )

    def _draw_tree(self, t, id):
        import matplotlib.pyplot as plt
        from sklearn import tree

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1)
        tree.plot_tree(t,
                       node_ids=True,
                       feature_names=self.feature_names,
                       class_names=self.target_class,
                       filled=False,
                       ax=ax)
        plt.tight_layout()
        plt.savefig(f'./{id}.pdf', dpi=300)

    def tweak(self, X, y, epsilon=0.1):
        """
        X (numpy ndarray): Features
        y (numpy ndarray of 1 dim): Class labels
        """
        self.epsilon = float(epsilon)
        if self.epsilon <= 0.0:
            self.logger.exception(
                "epsilon may not be negative. provided %d" % epsilon)
            raise ValueError(
                "epsilon may not be negative. provided %d" % epsilon
            )

        self.logger.info(
            "Retrieving the list of true negative instances")
        y_hat = self.model.predict(X)
        # select true negative instances from X
        y_diff = np.where((y==0) & (y==y_hat))

        self.logger.info("Creating the pool of workers")
        pool = mp.Pool()

        self.logger.info("Preparing the input to be sent to each worker of the pool")
        input_pairs = zip(range(0, y_diff[0].size), y_diff[0])
        # create inputs and assign them to the workers in parallel
        inputs = [(X[i], n, i, self.epsilon, self.model, self.paths)
                  for (n, i) in input_pairs]

        self.logger.info(
            "Computing all the possible epsilon-transformations in parallel")
        self.X_negatives_transformations = dict(
            pool.map(self.map_compute_epsilon_transformation, inputs))

        # For debugging TODO: remove this
        self._print_x_transformation()
        pass


    def map_compute_epsilon_transformation(self, instance):
        x, n, i, epsilon, model, paths = instance

        self.logger.info(
            "Computing all the possible epsilon-transformations for instance %d, %d" % (n, i))
        return (i, self.compute_epsilon_transformations_of_instance(x, n, i, epsilon, model, paths))

    def compute_epsilon_transformations_of_instance(self, x, n, i, epsilon, model, paths):
        x_transformations = {}
        path_conditions = {}
        tree_id = 0

        self.logger.info("Loop through all the decision trees of the ensemble for instance %d, %d" % (n, i))
        for decision_tree in tqdm(model.estimators_):
            r = decision_tree.predict(x.reshape(1, -1))
            y_hat_dt = model.classes_[int(r[0])]
            r_e = model.predict(x.reshape(1, -1))
            y_hat_ensemble = model.classes_[int(r_e[0])]

            if y_hat_dt == y_hat_ensemble:
                paths_dt = paths[tree_id]['paths']

                path_id = 0
                for path in paths_dt:
                    x_prime = self.compute_epsilon_transformation_path(x, epsilon, path, path_conditions)
                    r_prime = decision_tree.predict(x_prime.reshape(1, -1))
                    y_hat_prime_dt = model.classes_[int(r_prime[0])]

                    if y_hat_prime_dt == 1:
                        r_e_prime = model.predict(x_prime.reshape(1, -1))
                        y_hat_prime_ensemble = model.classes_[int(r_e_prime[0])]
                        if y_hat_prime_ensemble == 1:
                            candidate = (path_id, len(path), x_prime)
                            if tree_id in x_transformations:
                                x_transformations[tree_id].append(candidate)
                            else:
                                x_transformations[tree_id] = [candidate]

                    path_id += 1
            tree_id +=1
        return x_transformations



    def compute_epsilon_transformation_path(self, x, epsilon, path, path_conditions):
        """
        This function computes the epsilon transformation of an instance x
        according to the boolean conditions encoded in the specified path

            Args:
            x (ndarray): vector representing the instance x = (x_1, x_2, ..., x_n)

            epsilon (float): tolerance used to pass the tests encoded in path

            path (list(dict)): encoding of a root-to-leaf path of a decision tree as
            [(0, <dir>, theta_0), ..., (n-1, <dir>, theta_{n-1})]
            where each (i, <dir>, theta_i) encode a boolean condition as follows
            - if <dir> = "<=" then (i, "<=", theta_i) means that the (i+1)-th feature must be less than or equal to theta_i
            (x_{i+1} <= theta_i)
            - if <dir> = ">" then (i, ">", theta_i) means that the (i+1)-th feature must be greater than theta_i
            (x_{i+1} > theta_i)
            (Note: the discrepancy of the indices derives from the fact that features are 0-based indexed on the path,
            although usually they are referred using 1-based notation)

            Returns:
            tuple(x_prime, cost) where
            x_prime (pandas.Series): a transformation of the original instance x so that x_prime satisfies
            the conditions encoded in path with an epsilon tolerance
            For example, if x = (1.2, -3.7, 0.8) and path = [(0, <=, 1.5), (1, <=, -4)]
            x_prime = (
                1.2, -4-epsilon, 0.8)
            Indeed, the first boolean condition encoded in the path states that
            - (x_{0+1} <= 1.5) = (x_1 <= 1.5) Since x_1 = 1.2 this condition is already satisfied
            - (x_{1+1} <= -4) = (x_2 <= -4) Since x_2 = -3.7 this value must be changed accordingly
            so to satisfy the path, namely we set x_2 = -4-epsilon
            - Finally, since there is no condition for x_3, we let it as it is.
        """
        x_prime = x.copy()
        for condittion in path:
            feature = condittion['feature']
            direction = condittion['direction']
            threshold = condittion['threshold']
            feature_id = np.where(self.feature_names==feature)[0][0]
            cond = (feature_id, direction, threshold)

            # 1. if we have already examined this condition for this instance x then
            # we just retrieve the correct feature value for the transformed instance x'
            # 2. otherwise, we must compute the new feature value for the
            # transformed instance x'
            if cond in path_conditions:
                x_prime[feature_id] = path_conditions[cond]
            else:
                # Negative Direction Case: (x_i, theta_i, <=) ==> x_i must be less than or equal
                # to theta_i (x_i <= theta_i)
                if direction == "<=":
                    if x[feature_id] <= threshold:
                        pass
                    else:
                        x_prime[feature_id] = threshold - epsilon
                # Positive Direction Case: (x_i, theta_i, >) ==> x_i must be greater than
                # theta_i (x_i > theta_i)
                else:
                    if x[feature_id] > threshold:
                        pass
                    else:
                        x_prime[feature_id] = threshold + epsilon
                path_conditions[cond] = x_prime[feature_id]
        return x_prime

    def _print_x_transformation(self):
        for key in self.X_negatives_transformations:
            for tree_id in sorted(self.X_negatives_transformations[key]):
                for element in self.X_negatives_transformations[key][tree_id]:
                    path_id = element[0]
                    path_length = element[1]
                    x_prime = element[2]
                    print("%d \t %d \t %d \t %d \t %s\n" % (key, tree_id, path_id, path_length, '\t'.join([str(x) for x in x_prime])))

