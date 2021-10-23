""" Construct node types based on the config dict.

NOTE: do not remove unused imports, they are using in globals() call below

"""

__author__ = "Subhas K. Ghosh"
__copyright__ = "Copyright (C) 2021 GTM.ai"
__version__ = "1.0"

############## DO NOT REMOVE UNUSED IMPORTS ############
from core.nodes.common.generic import *
from core.nodes.extract.db.reader import *
from core.nodes.extract.file.csv_reader import *
from core.nodes.info.profiler import *
from core.nodes.load.db.writer import *
from core.nodes.load.file.csv_writer import *
from core.nodes.ml.classifier.rf import *
from core.nodes.ml.clustering.kmeans import *
from core.nodes.ml.processing.split import *
from core.nodes.ml.transformer.transformation import *
from core.nodes.node import *
from core.nodes.transform.preprocessing import *
############## DO NOT REMOVE UNUSED IMPORTS ############

class NodeFactory(object):
    """Construct a node by name type and return"""
    def __init__(self):
        self.logger = get_logger("NodeFactory")
        self.nodes = {
            'extract.csv': 'CSVFIleReaderNode',
            'extract.postgres': 'PostgresReaderNode',
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
            'load.csv': 'CSVFIleWriterNode',
            'info.cluster.profiler': 'ClusterProfiler',
            'info.df.profiler': 'DatFrameProfiler',
            'transform.dataframe.dropna': 'DataFrameDropNA',
            'transform.dataframe.impute.zero': 'DataFrameImputeZero',
            'transform.dataframe.convert.datetime': 'DataFrameConvertDateTime',
            'transform.dataframe.timediff': 'DataFrameAddTimediff',
            'transform.dataframe.project': 'DataFrameProject',
            'transform.dataframe.string.impute': 'DataFrameStringImpute',
            'transform.dataframe.conditional.string.impute': 'DataFrameConditionalStringImpute',
            'ml.train_test.split': 'TrainTestSplit',
            'ml.transform.RobustScaleTransformer': 'RobustScaleTransformer',
            'ml.transform.WoeTransformer': 'WoeTransformer',
            'ml.transform.ColumnTransformer': 'ColumnTransformerNode',
            'ml.classifier.RandomForest': 'RandomForest',
            'load.metadata.postgres': 'PostgresMetadataWriterNode',
            'query.metadata.postgres': 'PostgresQueryMetadataNode',
            'load.postgres': 'PostgresLoadDataFrameNode'
        }

    def get(self, name, parameter, input, output):
        if name in self.nodes:
            node = globals()[self.nodes[name]](name, parameter, input, output)
            return node
        else:
            self.logger.warn(f"Trying to create a new type of node: {name} that is not implemented yet! Defaulting to AbstructNode!")
            return AbstructNode(name, parameter, input, output)