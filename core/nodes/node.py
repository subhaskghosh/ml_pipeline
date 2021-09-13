""" Executable node base class.
This script defines the class that can be used for defining a node in the DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""


class AbstructNode:
    def __init__(self, name, parameter, input, output):
        self.name = name
        self.parameter = parameter
        self.input = input
        self.output = output

    def __repr__(self):
        return f"node({self.name},{self.parameter},{self.input},{self.output})"

    def __hash__(self):
        return hash(f"node({self.name},{self.parameter},{self.input},{self.output})")

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self.name == other.name and
                self.parameter == other.parameter and
                self.input == other.input and
                self.output == other.output
        )

    def __str__(self):
        import textwrap
        return f"Param: {textwrap.fill(str(self.parameter),40)},\n I: {self.input},\n O: {self.output}"

    def execute(self):
        print(f"Executing: {self.name} with input data {self.input_data}")

    def accept_delivery(self, input):
        self.input_data = input

class AbstructNodeResult:
    '''Result data type as outout from node execution
    Use type for passing the dict schema of the result
    if required
    '''
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.data = None

    def __repr__(self):
        return f"result({self.name},{self.type})"

    def __hash__(self):
        return hash(f"node({self.name},{self.type})")

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self.name == other.name and
                self.type == other.type
        )

    def __str__(self):
        return f"result({self.name},{self.type})"

    def update_payload(self,data):
        self.data = data

    def get_payload(self):
        return self.data