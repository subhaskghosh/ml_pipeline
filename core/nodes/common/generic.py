""" Executable node of generic type.
This script defines the class that can be used for defining a node in the DAG.
"""
__author__ = "Subhas K. Ghosh"
__version__ = "1.0"

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

class UnionColumnNames(AbstructNode):
    """Uppercase all column names"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        # Validate parameter
        if 'cached_variables' in self.parameter:
            self.cached_variables = self.parameter['cached_variables']
        else:
            raise NodeConfigurationError(
                'Variables not specified "{0}"'.format(parameter))

        if 'lhs_variable_name' in self.parameter:
            self.lhs_variable_name = self.parameter['lhs_variable_name']
        else:
            raise NodeConfigurationError(
                'Output Variable name not specified "{0}"'.format(parameter))

    def execute(self):
        variable_value = []
        for variable in self.cached_variables:
            variable_value += list(self.getFromCache(variable))
        self.addToCache(self.lhs_variable_name, variable_value)