""" Executable node of ML type.
This script defines the class that can be used for defining a node in the DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
from sklearn.model_selection import train_test_split
from core.error import NodeConfigurationError
from core.nodes.node import AbstructNode
import numpy as np

class TrainTestSplit(AbstructNode):
    """Train Test Split"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # Validate parameter

        if 'output_vars' in self.parameter:
            self.output_vars = self.parameter['output_vars']
        else:
            raise NodeConfigurationError(
                'output_vars not specified "{0}"'.format(parameter))

        if 'label' in self.parameter:
            self.label = self.parameter['label']
        else:
            raise NodeConfigurationError(
                'label column not specified "{0}"'.format(parameter))

        if 'data_columns' in self.parameter:
            self.data_columns = self.parameter['data_columns']
        else:
            raise NodeConfigurationError(
                'data columns not specified "{0}"'.format(parameter))

        if 'test_size' in self.parameter:
            self.test_size = self.parameter['test_size']
        else:
            raise NodeConfigurationError(
                'test data size not specified "{0}"'.format(parameter))

        if 'random_state' in self.parameter:
            self.random_state = self.parameter['random_state']
        else:
            self.random_state = 42

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

    def execute(self):
        self.label = self.getFromCache(self.label)
        self.data_columns = self.getFromCache(self.data_columns)
        data_df = self.getFromCache(self.input)

        X_arr = data_df[self.data_columns ].values
        y_arr = data_df[self.label].values

        X_train, X_test, y_train, y_test = train_test_split(X_arr,
                                                            y_arr,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)

        pos_train = np.count_nonzero(y_train) / y_train.shape[0]
        pos_test = np.count_nonzero(y_test) / y_test.shape[0]

        self.addToCache(self.output_vars[0], X_train)
        self.addToCache(self.output_vars[1], X_test)
        self.addToCache(self.output_vars[2], y_train)
        self.addToCache(self.output_vars[3], y_test)