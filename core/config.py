""" YAML to configuration.
This script defines the class that can be used for building a configuration object.
The config object can be feed into DAG to generate the pipeline.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""

import yaml

from core.logmanager import get_logger
from core.yaml_loader import SubstitutionLoader, user_params

class ConfigBuilder(object):
    """Reads a yaml config and returns a dict"""
    def __init__(self, path=None, params=None):
        self.logger = get_logger("ConfigBuilder")
        self.logger.info(f"Building from config path {path} with parameter {params}")
        self.config = None
        if params:
            for k,v in params.items():
                user_params[k] = v
        if path:
            try:
                self.stream = open(path, 'r')
                self.config = yaml.load(self.stream, Loader=SubstitutionLoader) # yaml.Loader
            except FileNotFoundError as e:
                self.logger.exception(
                    'Error opening Config "{0}"'.format(e))
        else:
            self.config = None

    def get(self):
        return self.config

    def show(self):
        self.logger.info(self.config)