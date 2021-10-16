""" Executable node of ML type.
This script defines the class that can be used for defining a node in the DAG.
"""
__author__ = "Subhas K. Ghosh"
__copyright__ = "Copyright (C) 2021 GTM.ai"
__version__ = "1.0"
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from core.error import NodeConfigurationError
from core.nodes.node import AbstructNode
import pickle

class KMeansClustering(AbstructNode):
    """KMeans Clustering"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # Validate parameter
        if 'n_clusters' in self.parameter:
            self.n_clusters = self.parameter['n_clusters']
        else:
            raise NodeConfigurationError(
                'Number of clusters not specified "{0}"'.format(parameter))

        if 'n_init' in self.parameter:
            self.n_init = self.parameter['n_init']
        else:
            raise NodeConfigurationError(
                'Number of time the k-means algorithm will be run with different centroid seeds is not specified "{0}"'.format(parameter))

        if 'random_state' in self.parameter:
            self.random_state = self.parameter['random_state']
        else:
            raise NodeConfigurationError(
                'random_state is not specified "{0}"'.format(parameter))

        if 'max_iter' in self.parameter:
            self.max_iter = self.parameter['max_iter']
        else:
            raise NodeConfigurationError(
                'max_iter is not specified "{0}"'.format(parameter))

        if 'algorithm' in self.parameter:
            self.algorithm = self.parameter['algorithm']
        else:
            raise NodeConfigurationError(
                'algorithm “auto”, “full”, or “elkan” not specified "{0}"'.format(parameter))

        if 'fit_df' in self.parameter:
            self.fit_df = self.parameter['fit_df']
        else:
            raise NodeConfigurationError(
                'Dataframe to fit not specified "{0}"'.format(parameter))

        if 'fit_df_columns' in self.parameter:
            self.fit_df_columns = self.parameter['fit_df_columns']
        else:
            raise NodeConfigurationError(
                'Dataframe columns to fit not specified "{0}"'.format(parameter))

        if 'from_variable' in self.parameter:
            self.from_variable = self.parameter['from_variable']

        if 'predict' in self.parameter:
            self.predict = self.parameter['predict']
        else:
            self.predict = None

        if 'update_df_with_prediction' in self.parameter:
            self.update_df_with_prediction = self.parameter['update_df_with_prediction']
        else:
            self.update_df_with_prediction = None

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

    def execute(self):
        if self.from_variable:
            self.fit_df_columns = self.getFromCache(self.fit_df_columns)
        df = self.getFromCache(self.fit_df)

        km = KMeans

        if self.mode == "save":
            km_obj = km(init='k-means++',
                        n_clusters=self.n_clusters,
                        n_init=self.n_init,
                        algorithm=self.algorithm,
                        max_iter=self.max_iter,
                        tol=1e-10,
                        random_state=self.random_state)
            X = df[self.fit_df_columns]
            km_obj.fit(X)
            with open(self.model_path, 'wb') as f:
                pickle.dump(km_obj, f)
        else:
            with open(self.model_path, 'rb') as f:
                km_obj = pickle.load(f)

        for k,v in self.predict.items():
            d = self.getFromCache(v['df'])
            c = self.getFromCache(v['cols'])
            tdf = km_obj.predict(d[c])
            self.addToCache(k, tdf)

        for k, v in self.update_df_with_prediction.items():
            d = self.getFromCache(v['df'])
            l = v['label']
            a = self.getFromCache(v['using'])
            d.loc[:, l] = a
            self.addToCache(k, d)

        results = {}
        if self.score:
            if 'silhouette_score' in self.score:
                sscore = self.score['silhouette_score']
                d = self.getFromCache(sscore['df'])
                l = sscore['label']
                c = self.getFromCache(sscore['cols'])

                score = silhouette_score(d[c], d[l])

                results['silhouette_score'] = score

            if 'calinski_harabasz_score' in self.score:
                sscore = self.score['calinski_harabasz_score']
                d = self.getFromCache(sscore['df'])
                l = sscore['label']
                c = self.getFromCache(sscore['cols'])

                score = calinski_harabasz_score(d[c], d[l])

                results['calinski_harabasz_score'] = score

            if 'davies_bouldin_score' in self.score:
                sscore = self.score['davies_bouldin_score']
                d = self.getFromCache(sscore['df'])
                l = sscore['label']
                c = self.getFromCache(sscore['cols'])

                score = davies_bouldin_score(d[c], d[l])

                results['davies_bouldin_score'] = score

            if 'update_cache' in self.score:
                self.addToCache(self.score['update_cache'], results)