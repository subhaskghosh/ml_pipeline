""" DAG executors.
"""
__author__ = "Subhas K. Ghosh"
__version__ = "1.0"

from core.logmanager import get_logger


class NullExecutor(object):
    '''An executor which runs no real things'''

class Executor(object):
    def __init__(self):
        self.logger = get_logger("Executor")

    def param(self, vertex):
        return vertex

    def execute(self, param):
        return param.execute()

    def report_start(self, vertices):
        self.logger.info(f'Starting: {vertices}')

    def report_running(self, vertices):
        self.logger.info(f'Currently running: {vertices}')

    def report_finish(self, vertices_result):
        for vertex, result in vertices_result:
            self.logger.info('Finished running {0}'.format(vertex.name))

    def deliver(self, vertex, result):
        vertex.accept_delivery(result)