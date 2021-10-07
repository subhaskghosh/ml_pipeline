from unittest import TestCase
import sklearn
import sklearn.datasets
import sklearn.ensemble

from src.teaft.tweaker import ActionableFeatureTweaker


class TestActionableFeatureTweaker(TestCase):

    def test_case_1(self):
        breast_cancer = sklearn.datasets.load_breast_cancer()

        print(breast_cancer.feature_names)
        print(breast_cancer.target_names)

        train, test, labels_train, labels_test = \
            sklearn.model_selection.train_test_split(breast_cancer.data, breast_cancer.target, train_size=0.80)

        rf = sklearn.ensemble.RandomForestClassifier(criterion='gini', n_estimators=5)

        rf.fit(train, labels_train)

        print(sklearn.metrics.accuracy_score(labels_test, rf.predict(test)))

        aft = ActionableFeatureTweaker(model = rf,
                                       feature_names = breast_cancer.feature_names,
                                       class_names = breast_cancer.target_names,
                                       target_class = 'malignant')
