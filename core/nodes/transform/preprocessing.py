""" Executable node all dataframe preprocessing.
This script defines the class that can be used for defining a node in the DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
from sklearn.preprocessing import PowerTransformer

from core.error import NodeConfigurationError
from core.nodes.node import AbstructNode
from sklearn.impute import KNNImputer, SimpleImputer
import pandas as pd
import numpy as np
import ast
import re

pd.options.mode.chained_assignment = None

class ColumnUppcaseNode(AbstructNode):
    """Uppercase all column names"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # Validate that inout and output exists
        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        df = self.getFromCache(self.input)
        df.columns = df.columns.str.upper()
        self.addToCache(self.output,df)

class DataFrameTruncateExtreme(AbstructNode):
    """Mask extreme values with np.nan"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # validate parameters
        if 'inplace' in self.parameter:
            self.inplace = self.parameter['inplace']
        else:
            self.inplace = False

        if 'columns' in self.parameter:
            self.columns = self.parameter['columns']
        else:
            raise NodeConfigurationError(
                'Column name(s) not specified "{0}"'.format(parameter))

        if 'conditions' in self.parameter:
            self.conditions = self.parameter['conditions']
        else:
            raise NodeConfigurationError(
                'Filter conditions not specified "{0}"'.format(parameter))

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        df = self.getFromCache(self.input)

        if not self.inplace:
            tmp = df[self.columns]
        else:
            tmp = df

        for condition_dict in self.conditions:
            column = list(condition_dict.keys())[0]
            criterias = list(condition_dict.values())[0]

            for criteria in criterias:
                c,v = criteria.split(' ')
                try:
                    v = ast.literal_eval(v)
                except ValueError:
                    v = str(v)
                except SyntaxError:
                    raise NodeConfigurationError(
                        'Malformed "{0}"'.format(criteria))
                if c == '>':
                    do_filter = tmp[column] > v
                    tmp[column] = np.where(do_filter, np.nan, tmp[column])
                elif c == '<':
                    do_filter = tmp[column] < v
                    tmp[column] = np.where(do_filter, np.nan, tmp[column])
                elif c == '>=':
                    do_filter = tmp[column] >= v
                    tmp[column] = np.where(do_filter, np.nan, tmp[column])
                elif c == '<=':
                    do_filter = tmp[column] <= v
                    tmp[column] = np.where(do_filter, np.nan, tmp[column])
                elif c == '==':
                    do_filter = tmp[column] == v
                    tmp[column] = np.where(do_filter, np.nan, tmp[column])

        if not self.inplace:
            self.addToCache(self.output,tmp)
        else:
            df = tmp
            self.addToCache(self.output,df)

class DataFrameStringFilter(AbstructNode):
    """Filter rows by conditions"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # validate parameters
        if 'inplace' in self.parameter:
            self.inplace = self.parameter['inplace']
        else:
            self.inplace = False

        if 'columns' in self.parameter:
            self.columns = self.parameter['columns']
        else:
            raise NodeConfigurationError(
                'Column name(s) not specified "{0}"'.format(parameter))

        if 'conditions' in self.parameter:
            self.conditions = self.parameter['conditions']
        else:
            raise NodeConfigurationError(
                'Filter conditions not specified "{0}"'.format(parameter))

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        df = self.getFromCache(self.input)

        if not self.inplace:
            tmp = df[self.columns]
        else:
            tmp = df

        for condition_dict in self.conditions:
            column = list(condition_dict.keys())[0]
            criterias = list(condition_dict.values())[0]

            for criteria in criterias:
                c,v = criteria.split(' ')
                try:
                    v = ast.literal_eval(v)
                except ValueError:
                    v = str(v)
                except SyntaxError:
                    raise NodeConfigurationError(
                        'Malformed "{0}"'.format(criteria))
                if c == 'like':
                    p = re.compile(v, flags=re.IGNORECASE)
                    tmp = tmp[[bool(p.search(x)) for x in tmp[column]]]
                elif c == '==':
                    tmp = tmp[tmp[column]==v]

        if not self.inplace:
            self.addToCache(self.output,tmp)
        else:
            df = tmp
            self.addToCache(self.output,df)

class DataFrameFilterKNNImputer(AbstructNode):
    """KNN Imputation"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)

        # validate parameters
        if 'using_column' in self.parameter:
            self.using_column = self.parameter['using_column']
        else:
            raise NodeConfigurationError(
                'Using Column name(s) not specified "{0}"'.format(parameter))

        if 'on_columns' in self.parameter:
            self.on_columns = self.parameter['on_columns']
        else:
            raise NodeConfigurationError(
                'On Column name(s) not specified "{0}"'.format(parameter))

        if 'n_neighbors' in self.parameter:
            self.n_neighbors = self.parameter['n_neighbors']
        else:
            self.n_neighbors = 5

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        df = self.getFromCache(self.input)

        for industry in df[self.using_column].unique():
            try:
                if(not industry):
                    continue

                to_inpute = df[df[self.using_column] == industry][self.on_columns]
                imputer = KNNImputer(n_neighbors=self.n_neighbors)
                imputed = imputer.fit_transform(to_inpute)
                imputed = pd.DataFrame(imputed, columns=self.on_columns)

                for column in self.on_columns:
                    df.loc[df[self.using_column] == industry, column] = imputed[column].values
            except Exception as e:
                print(f"Info: {e}")
                continue
        self.addToCache(self.output, df)

class DataFrameAppendColumn(AbstructNode):
    """Copy columns from one DF to another"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)

        # validate parameters
        if 'from_df' in self.parameter:
            self.from_df_name = self.parameter['from_df']
        else:
            raise NodeConfigurationError(
                'from_df_name not specified "{0}"'.format(parameter))

        if 'to_df' in self.parameter:
            self.to_df_name = self.parameter['to_df']
        else:
            raise NodeConfigurationError(
                'to_df not specified "{0}"'.format(parameter))

        if 'from_df_columns' in self.parameter:
            self.from_df_columns = self.parameter['from_df_columns']
        else:
            raise NodeConfigurationError(
                'from_df_columns not specified "{0}"'.format(parameter))

        if 'to_df_columns' in self.parameter:
            self.to_df_columns = self.parameter['to_df_columns']
        else:
            raise NodeConfigurationError(
                'to_df_columns not specified "{0}"'.format(parameter))

        if 'from_df_columns_from_variable' in self.parameter:
            self.from_df_columns_from_variable = self.parameter['from_df_columns_from_variable']
        else:
            self.from_df_columns_from_variable = None

        if 'to_df_columns_from_variable' in self.parameter:
            self.to_df_columns_from_variable = self.parameter['to_df_columns_from_variable']
        else:
            self.to_df_columns_from_variable = None

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        if self.from_df_columns_from_variable:
            self.from_df_columns = self.getFromCache(self.from_df_columns)

        if self.to_df_columns_from_variable:
            self.to_df_columns = self.getFromCache(self.to_df_columns)

        if len(self.from_df_columns) != len(self.to_df_columns):
            raise NodeConfigurationError(
                'Number of columns must be same')
        from_df = self.getFromCache(self.from_df_name)
        to_df = self.getFromCache(self.to_df_name)
        to_df[list(self.to_df_columns)] = from_df[list(self.from_df_columns)]
        self.addToCache(self.output, to_df)


class DataFrameColumnOneHotEncoding(AbstructNode):
    """One hot encode columns and append them back to df"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)

        # validate parameters
        if 'columns' in self.parameter:
            self.columns = self.parameter['columns']
        else:
            raise NodeConfigurationError(
                'columns not specified "{0}"'.format(parameter))

        if 'save_ohe_column_names_as' in self.parameter:
            self.save_ohe_column_names_as = self.parameter['save_ohe_column_names_as']
        else:
            self.save_ohe_column_names_as = 'ohe_cols'

        if 'from_variable' in self.parameter:
            self.from_variable = self.parameter['from_variable']

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        if self.from_variable:
            self.columns = self.getFromCache(self.columns)
        df = self.getFromCache(self.input)
        coded = pd.get_dummies(df[self.columns].astype(str))
        df[coded.columns] = coded.values
        self.addToCache(self.output, df)
        self.addToCache(self.save_ohe_column_names_as, coded.columns)


class DataFrameNumericalFilter(AbstructNode):
    """Filter rows by conditions"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # validate parameters
        if 'inplace' in self.parameter:
            self.inplace = self.parameter['inplace']
        else:
            self.inplace = False

        if 'columns' in self.parameter:
            self.columns = self.parameter['columns']
        else:
            raise NodeConfigurationError(
                'Column name(s) not specified "{0}"'.format(parameter))

        if 'conditions' in self.parameter:
            self.conditions = self.parameter['conditions']
        else:
            raise NodeConfigurationError(
                'Filter conditions not specified "{0}"'.format(parameter))

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        df = self.getFromCache(self.input)

        if not self.inplace:
            tmp = df[self.columns]
        else:
            tmp = df

        for condition_dict in self.conditions:
            column = list(condition_dict.keys())[0]
            criterias = list(condition_dict.values())[0]

            for criteria in criterias:
                c,v = criteria.split(' ')
                try:
                    v = ast.literal_eval(v)
                except ValueError:
                    v = str(v)
                except SyntaxError:
                    raise NodeConfigurationError(
                        'Malformed "{0}"'.format(criteria))
                if c == '>':
                    tmp = tmp[tmp[column] > v]
                elif c == '<':
                    tmp = tmp[tmp[column] < v]
                elif c == '>=':
                    tmp = tmp[tmp[column] >= v]
                elif c == '<=':
                    tmp = tmp[tmp[column] <= v]
                elif c == '!=':
                    tmp = tmp[tmp[column] != v]
                elif c == '==':
                    tmp = tmp[tmp[column] == v]


        if not self.inplace:
            self.addToCache(self.output,tmp)
        else:
            df = tmp
            self.addToCache(self.output,df)


class DataFrameComparatorFilter(AbstructNode):
    """Filter rows by comparing with list or dict"""

    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # validate parameters

        if 'conditions' in self.parameter:
            self.conditions = self.parameter['conditions']
        else:
            raise NodeConfigurationError(
                'Filter conditions not specified "{0}"'.format(parameter))

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        df = self.getFromCache(self.input)

        for condition_dict in self.conditions:
            column = condition_dict['column']
            comparator = condition_dict['comparator']
            compare_with = condition_dict['compare_with']

            if comparator == 'isin':
                df = df[df[column].isin(compare_with)]
            elif comparator == 'eq':
                df = df[df[column].eq(compare_with)]
            elif comparator == 'ne':
                df = df[df[column].ne(compare_with)]
            elif comparator == 'le':
                df = df[df[column].le(compare_with)]
            elif comparator == 'ge':
                df = df[df[column].ge(compare_with)]
            elif comparator == 'lt':
                df = df[df[column].lt(compare_with)]
            elif comparator == 'gt':
                df = df[df[column].gt(compare_with)]

        self.addToCache(self.output, df)

class DataFrameResetIndex(AbstructNode):
    """Reset Index"""

    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # validate parameters

        if 'drop' in self.parameter:
            self.drop = self.parameter['drop']

        if 'inplace' in self.parameter:
            self.inplace = self.parameter['inplace']

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        df = self.getFromCache(self.input)
        df.reset_index(drop=self.drop, inplace=self.inplace)
        self.addToCache(self.output, df)

class DataFrameSimpleImputation(AbstructNode):
    """Simple Imputation"""

    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # validate parameters
        if 'columns' in self.parameter:
            self.columns = self.parameter['columns']
        else:
            raise NodeConfigurationError(
                'Column name(s) not specified "{0}"'.format(parameter))

        if 'strategy' in self.parameter:
            self.strategy = self.parameter['strategy']
        else:
            raise NodeConfigurationError(
                'Strategy not specified "{0}"'.format(parameter))

        if 'from_variable' in self.parameter:
            self.from_variable = self.parameter['from_variable']

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        if self.from_variable:
            self.columns = self.getFromCache(self.columns)
        df = self.getFromCache(self.input)
        imputer = SimpleImputer(strategy=self.strategy)
        imputed = imputer.fit_transform(df[self.columns])
        data_imputed = pd.DataFrame(imputed, columns=self.columns)
        self.addToCache(self.output, data_imputed)

class DataFrameDetectOutlier(AbstructNode):
    """Detect outlier on numerical columns"""

    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # validate parameters
        # validate parameters
        if 'columns' in self.parameter:
            self.columns = self.parameter['columns']
        else:
            raise NodeConfigurationError(
                'Column name(s) not specified "{0}"'.format(parameter))

        if 'quantile' in self.parameter:
            self.quantile = self.parameter['quantile']
        else:
            raise NodeConfigurationError(
                'Quantile not specified "{0}"'.format(parameter))

        if 'interpolation' in self.parameter:
            self.interpolation = self.parameter['interpolation']
        else:
            raise NodeConfigurationError(
                'Interpolation method not specified "{0}"'.format(parameter))

        if 'outlier_column' in self.parameter:
            self.outlier_column = self.parameter['outlier_column']
        else:
            self.outlier_column = 'sum'

        if 'from_variable' in self.parameter:
            self.from_variable = self.parameter['from_variable']

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        if self.from_variable:
            self.columns = self.getFromCache(self.columns)
        df = self.getFromCache(self.input)

        qtile = df[self.columns].quantile(q=self.quantile, interpolation=self.interpolation)
        outlier = (df[self.columns] < qtile.loc[self.quantile[0], self.columns]) | (df[self.columns] > qtile.loc[self.quantile[1], self.columns])
        outlier = outlier.astype(int)
        outlier[self.outlier_column] = outlier.sum(axis=1)
        self.addToCache(self.output, outlier)

class DataFrameRemoveOutlier(AbstructNode):
    """Detect outlier on numerical columns"""

    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # validate parameters
        # validate parameters
        if 'outlier_column' in self.parameter:
            self.outlier_column = self.parameter['outlier_column']
        else:
            raise NodeConfigurationError(
                'outlier column name not specified "{0}"'.format(parameter))

        if 'outlier_condition' in self.parameter:
            self.outlier_condition = self.parameter['outlier_condition']
        else:
            raise NodeConfigurationError(
                'outlier condition not specified "{0}"'.format(parameter))

        if 'from_df' in self.parameter:
            self.from_df = self.parameter['from_df']
        else:
            raise NodeConfigurationError(
                'from_df not specified "{0}"'.format(parameter))

        if 'using_outlier_df' in self.parameter:
            self.using_outlier_df = self.parameter['using_outlier_df']
        else:
            raise NodeConfigurationError(
                'using_outlier_df not specified "{0}"'.format(parameter))

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        from_df = self.getFromCache(self.from_df)
        using_outlier_df = self.getFromCache(self.using_outlier_df)

        c, v = self.outlier_condition.split(' ')
        try:
            v = ast.literal_eval(v)
        except ValueError:
            v = str(v)
        except SyntaxError:
            raise NodeConfigurationError(
                'Malformed "{0}"'.format(self.outlier_condition))
        if c == '<':
            df = from_df[(using_outlier_df[self.outlier_column] < v)]
        elif c == '>':
            df = from_df[(using_outlier_df[self.outlier_column] > v)]
        elif c == '<=':
            df = from_df[(using_outlier_df[self.outlier_column] <= v)]
        elif c == '>=':
            df = from_df[(using_outlier_df[self.outlier_column] >= v)]

        self.addToCache(self.output, df)

class DataFramePowerTransform(AbstructNode):
    """Normalize"""

    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # validate parameters
        # validate parameters
        if 'columns' in self.parameter:
            self.columns = self.parameter['columns']
        else:
            raise NodeConfigurationError(
                'Column name(s) not specified "{0}"'.format(parameter))

        if 'method' in self.parameter:
            self.method = self.parameter['method']
        else:
            raise NodeConfigurationError(
                'Method not specified "{0}"'.format(parameter))

        if 'from_variable' in self.parameter:
            self.from_variable = self.parameter['from_variable']
        else:
            self.from_variable = None

        if 'reshape' in self.parameter:
            self.reshape = self.parameter['reshape']

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        if self.from_variable:
            self.columns = self.getFromCache(self.columns)

        df = self.getFromCache(self.input)
        pt = PowerTransformer(method=self.method)
        idf = df[self.columns]

        if self.reshape:
            idf = idf.values.reshape(self.reshape[0], self.reshape[1])

        tdf = pt.fit_transform(idf)
        ndf = pd.DataFrame(tdf, columns=self.columns, index=df.index)

        self.addToCache(self.output, ndf)

class DataFrameBooleanFilter(AbstructNode):
    """Filter rows by conditions"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # validate parameters
        if 'inplace' in self.parameter:
            self.inplace = self.parameter['inplace']
        else:
            self.inplace = False

        if 'condition_df' in self.parameter:
            self.condition_df = self.parameter['condition_df']
        else:
            raise NodeConfigurationError(
                'condition_df not specified "{0}"'.format(parameter))

        if 'columns' in self.parameter:
            self.columns = self.parameter['columns']
        else:
            raise NodeConfigurationError(
                'Column name(s) not specified "{0}"'.format(parameter))

        if 'conditions' in self.parameter:
            self.conditions = self.parameter['conditions']
        else:
            raise NodeConfigurationError(
                'Filter conditions not specified "{0}"'.format(parameter))

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

        if self.output == None:
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        df = self.getFromCache(self.input)
        condition_df = self.getFromCache(self.condition_df)

        if not self.inplace:
            tmp = df[self.columns]
        else:
            tmp = df

        for condition_dict in self.conditions:
            column = list(condition_dict.keys())[0]
            criterias = list(condition_dict.values())[0]

            for criteria in criterias:
                c,v = criteria.split(' ')
                try:
                    v = ast.literal_eval(v)
                except ValueError:
                    v = str(v)
                except SyntaxError:
                    raise NodeConfigurationError(
                        'Malformed "{0}"'.format(criteria))
                if c == '!=':
                    tmp = tmp[condition_df[column]!=v]
                elif c == '==':
                    tmp = tmp[condition_df[column]==v]

        if not self.inplace:
            self.addToCache(self.output,tmp)
        else:
            df = tmp
            self.addToCache(self.output,df)

