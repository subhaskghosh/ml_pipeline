""" Executable node of data profiler type.
This script defines the class that can be used for defining a node in the DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
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
        df = df[self.columns]
        profile = ProfileReport(df, title=title,
                                samples=None,
                                correlations=None,
                                missing_diagrams=None,
                                duplicates=None,
                                interactions=None,)

        if 'html' in self.report:
            is_html = self.report['html']
            path = self.report['path']
            if is_html:
                profile.to_file(f"{path}_{title}.html")

        if 'json' in self.report:
            is_json = self.report['json']
            path = self.report['path']
            if is_json:
                profile.to_file(f"{path}.json")