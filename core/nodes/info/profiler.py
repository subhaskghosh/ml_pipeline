""" Executable node of data profiler type.
This script defines the class that can be used for defining a node in the DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
import json
import pandas as pd
from core.error import NodeConfigurationError
from core.logmanager import get_logger
from core.nodes.node import AbstructNode
from pandas_profiling import ProfileReport

class ClusterProfiler(AbstructNode):
    """Clustering features profiler"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        self.logger = get_logger("DataFrameProfiler")
        # Validate parameter
        if 'report_path' in self.parameter:
            self.report_path = self.parameter['report_path']
        else:
            raise NodeConfigurationError(
                'Reporting csv path not specified "{0}"'.format(parameter))

        if 'columns' in self.parameter:
            self.columns = self.parameter['columns']
        else:
            raise NodeConfigurationError(
                'Columns not specified "{0}"'.format(parameter))

        if 'class_label' in self.parameter:
            self.class_label = self.parameter['class_label']
        else:
            raise NodeConfigurationError(
                'class_label Column not specified "{0}"'.format(parameter))

        if 'win_loss_column' in self.parameter:
            self.win_loss_column = self.parameter['win_loss_column']
        else:
            raise NodeConfigurationError(
                'win_loss_column not specified "{0}"'.format(parameter))

        if 'from_variable' in self.parameter:
            self.from_variable = self.parameter['from_variable']

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

    def execute(self):
        if self.from_variable:
            self.columns = self.getFromCache(self.columns)
        df = self.getFromCache(self.input)
        r = df.groupby([self.class_label, self.win_loss_column]).count()
        self.logger.info(f'\n\n {r.head()} \n\n')
        columns = []
        n_distinct = []
        p_distinct = []
        type = []
        n_missing = []
        n = []
        p_missing = []
        count = []
        memory_size = []
        n_zeros = []
        n_negative = []
        p_negative = []
        n_infinite = []
        mean = []
        std = []
        variance = []
        min = []
        max = []
        kurtosis = []
        skewness = []
        sum = []
        mad = []
        range = []
        prt5 = []
        prt25 = []
        prt50 = []
        prt75 = []
        prt95 = []
        iqr = []
        cv = []
        p_zeros = []
        p_infinite = []
        classification_label = []
        wl = []
        for win_loss in [True, False]:
            df_wl = df[df[self.win_loss_column] == win_loss]
            for label in [0,1,2]:
                df_wl_l = df_wl[df_wl[self.class_label]==label]
                df_wl_l = df_wl_l[self.columns]
                profile = ProfileReport(df_wl_l,
                                        samples=None,
                                        correlations=None,
                                        missing_diagrams=None,
                                        duplicates=None,
                                        interactions=None,
                                        progress_bar=False)

                res = json.loads(profile.to_json())
                if 'variables' in res:
                    variables = res['variables']
                    for k,v in variables.items():
                        if k in self.columns:
                            col_type = v['type']
                            if col_type == 'Numeric':
                                wl.append(win_loss)
                                classification_label.append(label)
                                columns.append(k)
                                n_distinct.append(v['n_distinct'])
                                p_distinct.append(v['p_distinct']*100.0)
                                type.append(col_type)
                                n_missing.append(v['n_missing'])
                                n.append(v['n'])
                                p_missing.append(v['p_missing']*100.0)
                                count.append(v['count'])
                                memory_size.append(v['memory_size'])
                                n_zeros.append(v['n_zeros'])
                                n_negative.append(v['n_negative'])
                                p_negative.append(v['p_negative']*100.0)
                                n_infinite.append(v['n_infinite'])
                                mean.append(v['mean'])
                                std.append(v['std'])
                                variance.append(v['variance'])
                                min.append(v['min'])
                                max.append(v['max'])
                                kurtosis.append(v['kurtosis'])
                                skewness.append(v['skewness'])
                                sum.append(v['sum'])
                                mad.append(v['mad'])
                                range.append(v['range'])
                                prt5.append(v['5%'])
                                prt25.append(v['25%'])
                                prt50.append(v['50%'])
                                prt75.append(v['75%'])
                                prt95.append(v['95%'])
                                iqr.append(v['iqr'])
                                cv.append(v['cv'])
                                p_zeros.append(v['p_zeros']*100.0)
                                p_infinite.append(v['p_infinite']*100.0)

        rdf = pd.DataFrame.from_dict({
            'Win-Loss': wl,
            'classification_label': classification_label,
            'Column Name': columns,
            'Distinct': n_distinct,
            'Distinct (%)': p_distinct,
            'Data Type': type,
            'Missing': n_missing,
            'Number of rows': n,
            'Missing (%)': p_missing,
            'Number of rows with value': count,
            'Memory used in bytes': memory_size,
            'Zeros': n_zeros,
            'Negative': n_negative,
            'Negative (%)': p_negative,
            'Infinite': n_infinite,
            'Mean': mean,
            'Standard deviation': std,
            'Variance': variance,
            'Minimum': min,
            'Maximum': max,
            'Kurtosis': kurtosis,
            'Skewness': skewness,
            'Sum': sum,
            'Median Absolute Deviation (MAD)': mad,
            'Range': range,
            '5-th percentile': prt5,
            'Q1': prt25,
            'Median': prt50,
            'Q3': prt75,
            '95-th percentile': prt95,
            'Interquartile range (IQR)': iqr,
            'Coefficient of variation (CV)': cv,
            'Zeros (%)': p_zeros,
            'Infinite (%)': p_infinite
        })

        rdf.to_csv(self.report_path)

class DatFrameProfiler(AbstructNode):
    """Dataframe profiler"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        self.logger = get_logger("DataFrameProfiler")
        # Validate parameter
        if 'report_path' in self.parameter:
            self.report_path = self.parameter['report_path']
        else:
            raise NodeConfigurationError(
                'Reporting html path not specified "{0}"'.format(parameter))

        if 'columns' in self.parameter:
            self.columns = self.parameter['columns']
        else:
            raise NodeConfigurationError(
                'Columns not specified "{0}"'.format(parameter))

        if 'remove_columns' in self.parameter:
            self.remove_columns = self.parameter['remove_columns']
        else:
            self.remove_columns = []

        if 'from_variable' in self.parameter:
            self.from_variable = self.parameter['from_variable']

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

    def execute(self):
        if self.from_variable:
            self.columns = self.getFromCache(self.columns)
        df = self.getFromCache(self.input)

        df = df[[item for item in self.columns if item not in self.remove_columns]]
        profile = ProfileReport(df,progress_bar=False)

        profile.to_file(self.report_path)