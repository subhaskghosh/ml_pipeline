""" Executable node local csv file write class.
This script defines the class that can be used for defining a node in the DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
from core.error import NodeConfigurationError
from core.nodes.node import AbstructNode
import pandas as pd
import json

class CSVFIleWriterNode(AbstructNode):
    """Read CSV file"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # validate that parameter has path
        if 'path' in self.parameter:
            self.path = self.parameter['path']
        else:
            raise NodeConfigurationError(
                'Incorrect CSV file path "{0}"'.format(parameter))
        # check if index is provided
        if 'index' in self.parameter:
            self.index = self.parameter['index']
        else:
            self.index = False

        # Validate that input definition exists
        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

    def execute(self):
        '''Write df to a csv file'''
        df = self.getFromCache(self.input)
        df.to_csv(self.path,index=self.index)