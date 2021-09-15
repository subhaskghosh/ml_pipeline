""" Executable node local csv file reader class.
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

class CSVFIleReaderNode(AbstructNode):
    """Read CSV file"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # validate that parameter has path
        if 'path' in self.parameter:
            self.path = self.parameter['path']
        else:
            raise NodeConfigurationError(
                'Incorrect CSV file path "{0}"'.format(parameter))
        # check if dtype is provided
        if 'dtype' in self.parameter:
            self.dtype_json_path = self.parameter['dtype']
        else:
            self.dtype_json_path = None

        # Validate that output definition exists
        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        '''read file from path and create a df with name specified by output'''
        if self.dtype_json_path:
            self.csv_dtype = json.load(self.dtype_json_path)
            df = pd.read_csv(self.path, dtype=self.csv_dtype)
        else:
            df = pd.read_csv(self.path)
        self.addToCache(self.output,df)