""" DAG executors.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""

class NullExecutor(object):
    '''An executor which runs no real things'''

class Executor(object):
    def param(self, vertex):
        return vertex

    def execute(self, param):
        return param.execute()

    def report_start(self, vertices):
        print('Starting:', vertices)

    def report_running(self, vertices):
        print('Currently running:', vertices)

    def report_finish(self, vertices_result):
        for vertex, result in vertices_result:
            print('Finished running {0}'.format(vertex.name))

    def deliver(self, vertex, result):
        vertex.accept_delivery(result)