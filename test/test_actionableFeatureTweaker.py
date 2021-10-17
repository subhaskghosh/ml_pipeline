import pickle
from unittest import TestCase
import sklearn
import sklearn.datasets
import sklearn.ensemble
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import Bunch

from core.nodes.ml.teaft.tweaker import ActionableFeatureTweaker


class TestActionableFeatureTweaker(TestCase):

    def test_case_1(self):
        breast_cancer = sklearn.datasets.load_breast_cancer()

        print(breast_cancer.feature_names)
        print(breast_cancer.target_names)

        train, test, labels_train, labels_test = \
            sklearn.model_selection.train_test_split(breast_cancer.data, breast_cancer.target, train_size=0.60)

        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=30)

        rf.fit(train, labels_train)

        print(sklearn.metrics.accuracy_score(labels_test, rf.predict(test)))

        aft = ActionableFeatureTweaker(model = rf,
                                       feature_names = breast_cancer.feature_names,
                                       class_names = breast_cancer.target_names,
                                       target_class = 'malignant')

        costfunc = 'euclidean_distance'

        [test_tweak, tweak_costs, tweak_signs] = aft.tweak(test, labels_test, epsilon=0.1, costfunc=costfunc)
        X_test_proba = rf.predict_proba(test)
        X_test_tweak_proba = rf.predict_proba(test_tweak)

        x_test_pos = X_test_proba[:, 1]
        x_test_tweak_pos = X_test_tweak_proba[:, 1]

        fig, axs = plt.subplots(1, 4, sharey=False, tight_layout=True, figsize=(20, 5))
        _ = axs[0].hist(x_test_pos, bins=40)
        _ = axs[1].hist(x_test_tweak_pos, bins=40)
        _ = axs[2].scatter(x_test_pos, x_test_tweak_pos)
        _ = axs[3].scatter(tweak_costs['id'].values.tolist(), tweak_costs[costfunc].values.tolist())

        tweak_signs = tweak_signs[list(breast_cancer.feature_names)]
        ax = tweak_signs.plot.box()
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.show()

    def test_case_2(self):
        data = pd.read_csv('/Users/ghoshsk/src/ds/ml_pipeline/test/resources/aft/csv/Epileptic Seizure Recognition.csv')
        data.drop('Unnamed',  axis=1, inplace=True)
        y = data.iloc[:, -1:].values.ravel()
        y[y > 1] = 0

        data.drop('y',  axis=1, inplace=True)

        target_names = ['epileptic', 'good']
        feature_names = np.array(data.columns.tolist())

        epileptic_seizure_recognition =  Bunch(
            data=data.values,
            target=y,
            target_names=target_names,
            feature_names=feature_names
        )

        print(epileptic_seizure_recognition.feature_names)
        print(epileptic_seizure_recognition.target_names)

        train, test, labels_train, labels_test = \
            sklearn.model_selection.train_test_split(epileptic_seizure_recognition.data,
                                                     epileptic_seizure_recognition.target,
                                                     train_size=0.80,
                                                     shuffle=True)

        rf = sklearn.ensemble.RandomForestClassifier(criterion='entropy', n_estimators=30)

        rf.fit(train, labels_train)

        print(sklearn.metrics.accuracy_score(labels_test, rf.predict(test)))
        costfunc = 'euclidean_distance'
        aft = ActionableFeatureTweaker(model = rf,
                                       feature_names = epileptic_seizure_recognition.feature_names,
                                       class_names = epileptic_seizure_recognition.target_names,
                                       target_class = 'epileptic')

        [test_tweak, tweak_costs, tweak_signs] = aft.tweak(test, labels_test, epsilon=0.1, costfunc=costfunc)
        X_test_proba = rf.predict_proba(test)
        X_test_tweak_proba = rf.predict_proba(test_tweak)

        x_test_pos = X_test_proba[:, 1]
        x_test_tweak_pos = X_test_tweak_proba[:, 1]

        fig, axs = plt.subplots(1, 4, sharey=False, tight_layout=True, figsize=(20, 5))
        _ = axs[0].hist(x_test_pos, bins=40)
        _ = axs[1].hist(x_test_tweak_pos, bins=40)
        _ = axs[2].scatter(x_test_pos, x_test_tweak_pos)
        _ = axs[3].scatter(tweak_costs['id'].values.tolist(), tweak_costs[costfunc].values.tolist())

        plt.show()

    def test_case_3(self):
        train = np.load('/Users/ghoshsk/src/ds/ml_pipeline/test/resources/dummy/model/train.npy')
        test = np.load('/Users/ghoshsk/src/ds/ml_pipeline/test/resources/dummy/model/test.npy')
        labels_train = np.load('/Users/ghoshsk/src/ds/ml_pipeline/test/resources/dummy/model/y_train.npy')
        labels_test = np.load('/Users/ghoshsk/src/ds/ml_pipeline/test/resources/dummy/model/y_test.npy')
        target_names = ['loss', 'win']
        feature_names = np.load('/Users/ghoshsk/src/ds/ml_pipeline/test/resources/dummy/model/features.npy')

        rf = sklearn.ensemble.RandomForestClassifier(criterion='entropy',
                                                     n_estimators=100,
                                                     max_depth=10)
        rf.fit(train, labels_train)
        print(sklearn.metrics.accuracy_score(labels_test, rf.predict(test)))
        costfunc = 'euclidean_distance'
        aft = ActionableFeatureTweaker(model=rf,
                                       feature_names=feature_names,
                                       class_names=target_names,
                                       target_class='win')

        [test_tweak, tweak_costs, tweak_signs] = aft.tweak(test, labels_test, epsilon=0.1, costfunc=costfunc)
        X_test_proba = rf.predict_proba(test)
        X_test_tweak_proba = rf.predict_proba(test_tweak)

        x_test_pos = X_test_proba[:, 1]
        x_test_tweak_pos = X_test_tweak_proba[:, 1]

        fig, axs = plt.subplots(1, 4, sharey=False, tight_layout=True, figsize=(20, 5))
        _ = axs[0].hist(x_test_pos, bins=40)
        _ = axs[1].hist(x_test_tweak_pos, bins=40)
        _ = axs[2].scatter(x_test_pos, x_test_tweak_pos)
        _ = axs[3].scatter(tweak_costs['id'].values.tolist(), tweak_costs[costfunc].values.tolist())

        plt.show()
