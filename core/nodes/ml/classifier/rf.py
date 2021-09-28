""" Executable node of ML type.
This script defines the class that can be used for defining a node in the DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, fbeta_score
from sklearn.model_selection import GridSearchCV

from core.error import NodeConfigurationError
from core.nodes.node import AbstructNode
import pickle

class RandomForest(AbstructNode):
    """KMeans Clustering"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # Validate parameter

        if 'search' in self.parameter:
            self.search = self.parameter['search']
        else:
            raise NodeConfigurationError(
                'search must be specified "{0}"'.format(parameter))

        if 'score' in self.parameter:
            self.score = self.parameter['score']
        else:
            self.score = None

        if 'mode' in self.parameter:
            self.mode = self.parameter['mode']
        else:
            raise NodeConfigurationError(
                'Mode as save or load must be specified "{0}"'.format(parameter))

        if 'model_path' in self.parameter:
            self.model_path = self.parameter['model_path']
        else:
            raise NodeConfigurationError(
                'Model path must be specified "{0}"'.format(parameter))

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

    def execute(self):
        X_train_T = self.getFromCache(self.input[0])
        X_test_T = self.getFromCache(self.input[1])
        y_train = self.getFromCache(self.input[2])
        y_test = self.getFromCache(self.input[3])

        rf = RandomForestClassifier(n_jobs=-1)


        if self.mode == "save":
            rf_param_grid = {'criterion': self.search['criterion'],
                             'n_estimators': self.search['n_estimators'],
                             'max_depth': self.search['max_depth'],
                             'min_samples_split': self.search['min_samples_split'],
                             'max_features': self.search['max_features']
                             }
            clf_rf = GridSearchCV(rf, rf_param_grid, cv=self.search['cross_validation'],n_jobs=-1)

            clf_rf.fit(X_train_T, y_train)

            with open(self.model_path, 'wb') as f:
                pickle.dump(clf_rf, f)
        else:
            with open(self.model_path, 'rb') as f:
                clf_rf = pickle.load(f)

        y_score_train = clf_rf.predict(X_train_T)
        y_score_test = clf_rf.predict(X_test_T)

        results = {}
        results['best_parameters'] = clf_rf.best_params_

        if self.score:
            if 'balanced_accuracy_score' in self.score:
                bac_train = balanced_accuracy_score(y_train, y_score_train)
                bac_test = balanced_accuracy_score(y_test, y_score_test)
                results['balanced_accuracy_score'] = {"train": bac_train, "test": bac_test}

            if 'fbeta_score' in self.score:
                sscore = self.score['fbeta_score']
                fbeta_train = fbeta_score(y_train, y_score_train, beta=sscore['beta'],average=sscore['average'])
                fbeta_test = fbeta_score(y_test, y_score_test, beta=sscore['beta'],average=sscore['average'])
                results['f0.5_score'] = {"train": fbeta_train, "test": fbeta_test}

            if 'importance' in self.score:
                sscore = self.score['importance']
                ml_cols = self.getFromCache(sscore['columns'])
                importances = list(zip(ml_cols, clf_rf.best_estimator_.feature_importances_))
                results['importance'] = sorted(importances, key=lambda x: x[1] * -1)

            if 'update_cache' in self.score:
                self.addToCache(self.score['update_cache'], results)