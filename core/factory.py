""" Construct node types based on the config dict.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
from core.nodes.common.generic import *
from core.nodes.extract.file.csv_reader import *
from core.nodes.load.file.csv_writer import CSVFIleWriterNode
from core.nodes.ml.clustering.kmeans import KMeansClustering
from core.nodes.node import AbstructNode
from core.nodes.transform.preprocessing import *

class NodeFactory(object):
    """Construct a node by name type and returns"""
    def __init__(self):
        self.nodes = {
            'extract.csv': 'CSVFIleReaderNode',
            'transform.column_name.upper': 'ColumnUppcaseNode',
            'transform.dataframe.truncate': 'DataFrameTruncateExtreme',
            'transform.dataframe.filter.string': 'DataFrameStringFilter',
            'transform.dataframe.knnimpute': 'DataFrameFilterKNNImputer',
            'transform.dataframe.append.column': 'DataFrameAppendColumn',
            'transform.dataframe.one_hot_encoding': 'DataFrameColumnOneHotEncoding',
            'transform.dataframe.filter.numerical': 'DataFrameNumericalFilter',
            'transform.dataframe.filter.compare': 'DataFrameComparatorFilter',
            'common.add.variable': 'AddVariables',
            'transform.dataframe.reset_index': 'DataFrameResetIndex',
            'transform.simple.imputation': 'DataFrameSimpleImputation',
            'transform.detect.outlier': 'DataFrameDetectOutlier',
            'transform.remove.outlier': 'DataFrameRemoveOutlier',
            'transform.dataframe.powertransform': 'DataFramePowerTransform',
            'transform.dataframe.boolean.filter': 'DataFrameBooleanFilter',
            'common.union.columns': 'UnionColumnNames',
            'ml.clustering.kmeans': 'KMeansClustering',
            'transform.dataframe.concat': 'DataFrameConcat',
            'load.csv': 'CSVFIleWriterNode'
        }

    def get(self, name, parameter, input, output):
        if name in self.nodes:
            node = globals()[self.nodes[name]](name, parameter, input, output)
            return node
        else:
            return AbstructNode(name, parameter, input, output)