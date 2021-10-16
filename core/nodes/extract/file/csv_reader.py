""" Executable node local csv file reader class.
This script defines the class that can be used for defining a node in the DAG.
"""
__author__ = "Subhas K. Ghosh"
__copyright__ = "Copyright (C) 2021 GTM.ai"
__version__ = "1.0"
from core.error import NodeConfigurationError
from core.logmanager import get_logger
from core.nodes.node import AbstructNode
import pandas as pd
import json

class CSVFIleReaderNode(AbstructNode):
    """Read CSV file"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        self.logger = get_logger("CSVFIleReaderNode")
        # validate that parameter has path
        if 'path' in self.parameter:
            self.path = self.parameter['path']
        else:
            self.logger.exception('Incorrect CSV file path "{0}"'.format(parameter))
            raise NodeConfigurationError(
                'Incorrect CSV file path "{0}"'.format(parameter))
        # check if dtype is provided
        if 'dtype' in self.parameter:
            self.dtype_json_path = self.parameter['dtype']
        else:
            self.dtype_json_path = None

        # Validate that output definition exists
        if self.output == None:
            self.logger.exception('Output can not be None')
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        '''read file from path and create a df with name specified by output'''
        self.logger.info(f'Reading CSV from {self.path}')
        if self.dtype_json_path:
            self.csv_dtype = json.load(self.dtype_json_path)
            df = pd.read_csv(self.path, dtype=self.csv_dtype, low_memory=False)
        else:
            df = pd.read_csv(self.path, low_memory=False)
        self.addToCache(self.output,df)