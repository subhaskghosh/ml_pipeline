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
from core.nodes.node import AbstructNode
from pandas_profiling import ProfileReport

class DataFrameProfiler(AbstructNode):
    """KMeans Clustering"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # Validate parameter
        if 'report' in self.parameter:
            self.report = self.parameter['report']
        else:
            raise NodeConfigurationError(
                'Reporting requirements not specified "{0}"'.format(parameter))

        if 'columns' in self.parameter:
            self.columns = self.parameter['columns']
        else:
            raise NodeConfigurationError(
                'Columns not specified "{0}"'.format(parameter))

        if 'from_variable' in self.parameter:
            self.from_variable = self.parameter['from_variable']

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

    def execute(self):
        title = f"Profiling Report - {self.columns}"
        if self.from_variable:
            self.columns = self.getFromCache(self.columns)
        df = self.getFromCache(self.input)
        r = df.groupby(['classification_label', 'DNA_STD_DC_END_RESULT']).count()
        print(r)
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
        DNA_STD_DC_END_RESULT = []
        for win_loss in [True, False]:
            df_wl = df[df['DNA_STD_DC_END_RESULT'] == win_loss]
            for label in [0,1,2]:
                df_wl_l = df_wl[df_wl['classification_label']==label]
                df_wl_l = df_wl_l[self.columns]
                profile = ProfileReport(df_wl_l, title=title,
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
                                DNA_STD_DC_END_RESULT.append(win_loss)
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
            'DNA_STD_DC_END_RESULT': DNA_STD_DC_END_RESULT,
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

        rdf.to_csv('/Users/ghoshsk/src/ds/ml_pipeline/test/resources/dummy/csv_output/profile.csv')

