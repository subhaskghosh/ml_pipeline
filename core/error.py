__author__ = "Subhas K. Ghosh"
__version__ = "1.0"

class DAGVertexNotFoundError(Exception):
    '''Exception when vertex not found'''


class DAGEdgeNotFoundError(Exception):
    '''Exception when edge not found'''


class DAGCycleError(Exception):
    '''Exception when cycle detected'''


class VertexExecutionError(Exception):
    '''Exception in vertex execution'''

class InvalidDBConfigError(Exception):
    '''Invalid Data Base Configuration Error'''

class ConfigNotFoundError(Exception):
    '''Invalid Configuration Error'''

class NodeConfigurationError(Exception):
    '''Invalid node configuration error'''

class CacheViolationError(Exception):
    '''Invalid cache access'''

class NodeDBError(Exception):
    '''Database errors'''