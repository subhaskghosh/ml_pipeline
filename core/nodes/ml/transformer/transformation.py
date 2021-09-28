""" Executable node all dataframe transformation.
This script defines the class that can be used for defining a node in the DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from core.error import NodeConfigurationError
from core.nodes.ml.transformer.woe import Woe
from core.nodes.node import AbstructNode

class RobustScaleTransformer(AbstructNode):
    """Scale transformer"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # Validate parameter

        if 'with_centering' in self.parameter:
            self.with_centering = self.parameter['with_centering']
        else:
            raise NodeConfigurationError(
                'with_centering not specified "{0}"'.format(parameter))

        if 'with_scaling' in self.parameter:
            self.with_scaling = self.parameter['with_scaling']
        else:
            raise NodeConfigurationError(
                'with_scaling not specified "{0}"'.format(parameter))

        if 'quantile_range' in self.parameter:
            self.quantile_range = self.parameter['quantile_range']
        else:
            raise NodeConfigurationError(
                'quantile_range not specified "{0}"'.format(parameter))

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        robust_scale_transformer = RobustScaler(with_centering=self.with_centering,
                                                with_scaling=self.with_scaling,
                                                quantile_range=(self.quantile_range[0], self.quantile_range[1]))
        self.addToCache(self.output, robust_scale_transformer)


class WoeTransformer(AbstructNode):
    """Weight of evidence transformer"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # Validate parameter

        if 'columns' in self.parameter:
            self.columns = self.parameter['columns']
        else:
            raise NodeConfigurationError(
                'columns not specified "{0}"'.format(parameter))

        if 'missing_val' in self.parameter:
            self.missing_val = self.parameter['missing_val']
        else:
            raise NodeConfigurationError(
                'missing_val not specified "{0}"'.format(parameter))

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        self.columns = self.getFromCache(self.columns)
        woe_transformer = Woe(col_names=self.columns)
        self.addToCache(self.output, woe_transformer)

class ColumnTransformerNode(AbstructNode):
    """Column Transformer"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # Validate parameter

        if 'pipeline' in self.parameter:
            self.pipeline = self.parameter['pipeline']
        else:
            raise NodeConfigurationError(
                'pipeline not specified "{0}"'.format(parameter))

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        pipeline = []
        start = 0
        for p in self.pipeline:
            type = p['type']
            transformer = self.getFromCache(p['transformer'])
            columns = self.getFromCache(p['columns'])

            pipeline.append((type,transformer,list(range(start, start+len(columns)))))
            start = start+len(columns)

        column_transformer = ColumnTransformer(pipeline)

        X_train = self.getFromCache(self.input[0])
        y_train = self.getFromCache(self.input[1])
        column_transformer.fit(X_train, y_train)
        X_train_T = column_transformer.transform(X_train)

        X_test = self.getFromCache(self.input[2])
        X_test_T = column_transformer.transform(X_test)

        self.addToCache(self.output[0], X_train_T)
        self.addToCache(self.output[1], X_test_T)