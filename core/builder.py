""" Builder class for DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""

from core.config import ConfigBuilder
from core.dag import Dag
from core.error import ConfigNotFoundError
from core.factory import NodeFactory
from core.logmanager import get_logger


class DAGBuilder(object):
    """Uses config builder to get a dict and
    traverses through the dict to generate an
    executable dag
    """
    def __init__(self, path=None, param=None):
        self.logger = get_logger("DAGBuilder")
        self.dag = None
        self.cb = ConfigBuilder(path, param)
        self.config = self.cb.get()
        self.nf = NodeFactory()
        nodes = []

        # Traverse through the config dict and create node objects
        if self.config:
            header = self.config[1]['pipelineMeta']
            stages = self.config[2]['stages']
            self.dag = Dag(meta=header)
            for stage in stages:
                name = list(stage.keys())[0]
                value = list(stage.values())[0]
                parameter = value['parameter']
                input = value['input']
                output = value['output']
                node = self.nf.get(name,parameter,input,output)

                # Give cache access to the node
                node.add_cache(self.dag.getCache())
                nodes.append(node)
                self.dag.add_vertex(node)

            # Now add the edges to the dag
            index = 0
            u = nodes[index]
            for index in range(1,len(nodes)):
                v = nodes[index]
                self.dag.add_edge(u,v)
                u = v
        else:
            self.logger.exception('Config "{0}" not specified or Null'.format(path))
            raise ConfigNotFoundError(
                'Config "{0}" not specified or Null'.format(path))

    def get(self):
        return self.dag

    def show(self):
        self.cb.show()