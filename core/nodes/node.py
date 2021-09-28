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
        pass

    def add_cache(self, cache):
        self.cache = cache

    def addToCache(self,k, v):
        self.cache.update(k, v)

    def getFromCache(self,k):
        return self.cache.get(k)

    def accept_delivery(self, data):
        pass
