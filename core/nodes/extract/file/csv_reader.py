""" Executable node local csv file reader class.
This script defines the class that can be used for defining a node in the DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
from core.error import NodeConfigurationError
from core.nodes.node import AbstructNode, AbstructNodeResult
import pandas as pd

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

    def execute(self):
        '''read file from path and create a df with name specified by output'''
        df = pd.read_csv(self.path)
        result = AbstructNodeResult(self.output,type(pd.DataFrame()))
        result.update_payload(df)
        return result