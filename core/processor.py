""" DAG processor - sequential for now. While some things can run in parallel
it might be hard to semantically capture that in YAML
"""
__author__ = "Subhas K. Ghosh"
__copyright__ = "Copyright (C) 2021 GTM.ai"
__version__ = "1.0"

from core.error import VertexExecutionError
from core.logmanager import get_logger


class NullProcessor(object):
    '''A processor which ignores all the execution'''

    def process(self, vertices_with_param, _):
        '''Return all vertices with result None'''
        return [(vtx, None) for vtx, _ in vertices_with_param]


class Processor(object):
    '''A processor which will run the executions in sequence'''
    def __init__(self):
        self.logger = get_logger("Executor")

    def process(self, vertices_with_param, execute_func):
        '''Process vertices in sequence'''
        results = []
        for vtx, param in vertices_with_param:
            try:
                result = execute_func(param)
            except Exception as e:
                self.logger.exception('Vertex "{0}" execution error: {1}'.format(vtx, e))
                raise VertexExecutionError(
                    'Vertex "{0}" execution error: {1}'.format(vtx, e))
            results.append((vtx, result))
        return results
