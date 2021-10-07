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
import logging

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
        self.logger = get_logger("ActionableFeatureTweaker")
        self.feature_names = feature_names
        self.class_names = class_names
        self.target_class = target_class
        self.precision = 3

        # Model is an instance of
        if not (isinstance(model, DecisionTreeClassifier) or
                isinstance(model, GradientBoostingClassifier) or
                isinstance(model, RandomForestClassifier)):
            self.logger.exception("Model must be one of DecisionTreeClassifier, GradientBoostingClassifier, or RandomForestClassifier")
            raise TypeError(
                "Model must be one of DecisionTreeClassifier, GradientBoostingClassifier, or RandomForestClassifier"
            )

        # Must have features that match with the model
        if self.feature_names is not None:
            if len(self.feature_names) != model.n_features_in_:
                self.logger.exception(
                    "Length of feature_names, %d does not match number of features, %d"
                    % (len(self.feature_names), model.n_features_in_))
                raise ValueError(
                    "Length of feature_names, %d does not match number of features, %d"
                    % (len(self.feature_names), model.n_features_in_)
                )
        else:
            self.logger.exception(
                "Feature names must be provided!")
            raise ValueError(
                "Feature names must be provided!"
            )

        # Must have classes that match with the model
        if self.class_names is not None:
            if len(self.class_names) != model.n_classes_:
                self.logger.exception(
                    "Length of classes, %d does not match number of classes, %d"
                    % (len(self.class_names), model.n_classes_))
                raise ValueError(
                    "Length of classes, %d does not match number of classes, %d"
                    % (len(self.class_names), model.n_classes_)
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
        self._forest_from_model(model)

        # get all positive paths for the target class
        self.paths = self._get_positive_paths()

    def _forest_from_model(self, model):
        self.forest = []
        if isinstance(model, DecisionTreeClassifier):
            self.n_forest = 1
            tree = self._tree_from_decision_tree(model)
            self.forest.append(tree)
        else:
            self.n_forest = model.n_estimators
            for index in range(0, model.n_estimators):
                decision_tree = model.estimators_[index]
                # added for debugging
                # self._draw_tree(decision_tree, index)
                tree = self._tree_from_decision_tree(decision_tree)
                self.forest.append(tree)

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
        positive_paths = []
        for tree in self.forest:
            paths = self._get_positive_path_on_dt(tree)
            positive_paths.append({'tree': tree, 'paths': paths})

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