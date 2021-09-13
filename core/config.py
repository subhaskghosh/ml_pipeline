""" JSON to configuration.
This script defines the class that can be used for building a configuration object.
The config object can be feed into DAG to generate the pipeline.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""

import yaml

from core.dag import Dag
from core.error import ConfigNotFoundError
from core.nodes.extract.file.csv_reader import CSVFIleReaderNode
from core.nodes.node import AbstructNode
from core.nodes.transform.preprocessing import ColumnUppcaseNode, DataFrameTruncateExtreme


class ConfigBuilder(object):
    """Reads a yaml config and returns a dict"""
    def __init__(self, path=None):
        self.config = None
        if path:
            try:
                self.stream = open(path, 'r')
                self.config = yaml.load(self.stream, yaml.Loader)
            except FileNotFoundError as e:
                print(
                    'Error opening Config "{0}"'.format(e))
        else:
            self.config = None

    def get(self):
        return self.config

    def show(self):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)

class DAGBuilder(object):
    """Uses config builder to get a dict and
    traverses through the dict to generate an
    executable dag
    """
    def __init__(self, path=None):
        self.dag = None
        self.cb = ConfigBuilder(path)
        self.config = self.cb.get()
        self.nf = NodeFactory()
        nodes = []

        # Traverse through the config dict and create node objects
        if self.config:
            stages = self.config[2]['stages']
            self.dag = Dag()
            for stage in stages:
                name = list(stage.keys())[0]
                value = list(stage.values())[0]
                parameter = value['parameter']
                input = value['input']
                output = value['output']
                node = self.nf.get(name,parameter,input,output)
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
            raise ConfigNotFoundError(
                'Config "{0}" not specified or Null'.format(path))

    def get(self):
        return self.dag

class NodeFactory(object):
    """Construct a node by name type and returns"""
    def __init__(self):
        pass

    def get(self, name, parameter, input, output):
        if name == 'extract.csv':
            return CSVFIleReaderNode(name, parameter, input, output)
        elif name == 'transform.column_name.upper':
            return ColumnUppcaseNode(name, parameter, input, output)
        elif name == 'transform.dataframe.truncate':
            return DataFrameTruncateExtreme(name, parameter, input, output)
        else:
            return AbstructNode(name, parameter, input, output)