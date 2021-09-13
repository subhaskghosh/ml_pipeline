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