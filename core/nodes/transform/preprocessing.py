""" Executable node all dataframe preprocessing.
This script defines the class that can be used for defining a node in the DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
from core.error import NodeConfigurationError
from core.nodes.node import AbstructNode, AbstructNodeResult
import pandas as pd
import numpy as np
import ast

pd.options.mode.chained_assignment = None
class ColumnUppcaseNode(AbstructNode):
    """Read CSV file"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)

    def execute(self):
        df = self.input_data.get_payload()
        df.columns = df.columns.str.upper()
        result = AbstructNodeResult(self.output, type(pd.DataFrame()))
        result.update_payload(df)
        return result

class DataFrameTruncateExtreme(AbstructNode):
    """Mask extreme values with nan"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # validate parameters
        if 'inplace' in self.parameter:
            self.inplace = self.parameter['inplace']
        else:
            self.inplace = False

        if 'column' in self.parameter:
            self.column = self.parameter['column']
        else:
            raise NodeConfigurationError(
                'Column name not specified "{0}"'.format(parameter))

        if 'conditions' in self.parameter:
            self.conditions = self.parameter['conditions']
        else:
            raise NodeConfigurationError(
                'Filter conditions not specified "{0}"'.format(parameter))

    def execute(self):
        df = self.input_data.get_payload()

        if not self.inplace:
            tmp = df[[self.column]]
        else:
            tmp = df

        for condition in self.conditions:
            c,v = condition.split(' ')
            try:
                v = ast.literal_eval(v)
            except ValueError:
                v = str(v)
            except SyntaxError:
                raise NodeConfigurationError(
                    'Malformed "{0}"'.format(condition))
            if c == '>':
                do_filter = tmp[self.column] > v
                tmp[self.column] = np.where(do_filter, np.nan, tmp[self.column])
            elif c == '<':
                do_filter = tmp[self.column] < v
                tmp[self.column] = np.where(do_filter, np.nan, tmp[self.column])
            elif c == '>=':
                do_filter = tmp[self.column] >= v
                tmp[self.column] = np.where(do_filter, np.nan, tmp[self.column])
            elif c == '<=':
                do_filter = tmp[self.column] <= v
                tmp[self.column] = np.where(do_filter, np.nan, tmp[self.column])
            elif c == '==':
                do_filter = tmp[self.column] == v
                tmp[self.column] = np.where(do_filter, np.nan, tmp[self.column])

        if not self.inplace:
            result = AbstructNodeResult(self.output, type([]))
            result.update_payload([df,tmp])
            return result
        else:
            df = tmp
            result = AbstructNodeResult(self.output, type(pd.DataFrame()))
            result.update_payload(df)
            return result