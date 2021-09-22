""" Executable node of ML type.
This script defines the class that can be used for defining a node in the DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
            km_obj = km(n_clusters=self.n_clusters, n_init=self.n_init, algorithm=self.algorithm)
            km_obj.fit(df[self.fit_df_columns])
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

        if self.score:
            d = self.getFromCache(self.score['df'])
            l = self.score['label']
            c = self.getFromCache(self.score['cols'])
            score = silhouette_score(d[c], d[l])
            print(f'Score: {score}')
            self.addToCache('score', score)