""" Executable node of generic type.
This script defines the class that can be used for defining a node in the DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
from core.error import NodeConfigurationError
from core.nodes.node import AbstructNode

class AddVariables(AbstructNode):
    """Uppercase all column names"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # Validate parameter
        if 'variables' in self.parameter:
            self.variables = self.parameter['variables']
        else:
            raise NodeConfigurationError(
                'Variables not specified "{0}"'.format(parameter))

    def execute(self):
        for variable in self.variables:
            variable_name = list(variable.keys())[0]
            variable_value = list(variable.values())[0]
            self.addToCache(variable_name, variable_value)
