""" JSON to configuration.
This script defines the class that can be used for building a configuration object.
The config object can be feed into DAG to generate the pipeline.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""

import yaml

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