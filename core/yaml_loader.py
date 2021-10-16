""" Identifies custom tags in YAML template and substitutes them.
1. Returns a stricter format - OtrderedDict instead of plain python dict
2. Supports rendering Date, UUID, Var - simple rendering for now
3. Added supports for resolving password with !Password tag
Based on ideas from:
Emrichen – Template engine for YAML & JSON: https://github.com/con2/emrichen#emrichen--template-engine-for-yaml--json
This script defines the class that can be used for building a configuration object.
The config object can be feed into DAG to generate the pipeline.
"""
__author__ = "Subhas K. Ghosh"
__copyright__ = "Copyright (C) 2021 GTM.ai"
__version__ = "1.0"

from typing import Any, Dict, Tuple, Type
from collections import OrderedDict
import yaml

tag_registry = {}
user_params = {}

class BaseMeta(type):
    def __new__(meta: Type['BaseMeta'], name: str, bases, class_dict: Dict[str, Any]):
        cls = type.__new__(meta, name, bases, class_dict)
        if name[0] != '_' and name != 'BaseTag':
            tag_registry[name] = cls
        return cls

class BaseTag(metaclass=BaseMeta):
    __slots__ = ['data']
    value_types: Tuple[Type, ...] = (str,)

    def __init__(self, data) -> None:
        self.data = data
        if not isinstance(data, self.value_types):
            raise TypeError(f'{self}: data not of valid type (valid types are {self.value_types}')

    def __str__(self) -> str:
        return f'{self.__class__.__name__}.{self.data!r}'

    def resolve(self) -> str: pass

class UUID(BaseTag):
    """
    example: !UUID
    description: Replaced with generated UUID.
    """

    def resolve(self):
        import uuid
        return f'{str(uuid.uuid4())}'

class Date(BaseTag):
    """
    arguments: Date string or function
    example: !Date today or !Date %Y%m%d
    description: renders date.
    """

    def resolve(self):
        from datetime import datetime
        if self.data == 'today':
            return f'{datetime.today()}'
        else:
            return f"{datetime.strptime(self.data, '%Y%m%d')}"

class Var(BaseTag):
    """
    arguments: Variable name
    example: !Var image_name
    description: Replaced with the value of the variable.
    """

    def resolve(self):
        if self.data in user_params:
            return f"{user_params[self.data]}"
        else:
            return f"UNDEFINED"

def construct_tagless_yaml(loader: yaml.Loader, node: yaml.Node):
    # From yaml.constructor.BaseConstructor#construct_object
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    elif isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    raise NotImplementedError('invalid node')


def construct_tagged_object(loader: yaml.Loader, node: yaml.Node) -> BaseTag:
    name = node.tag.lstrip('!')
    if name in tag_registry:
        tag = tag_registry[name]
        data = construct_tagless_yaml(loader, node)
        return (tag(data).resolve())

class SubstitutionLoader(yaml.SafeLoader):
    def __init__(self, stream) -> None:
        super().__init__(stream)
        self.add_tag_constructors()

    def add_tag_constructors(self) -> None:
        self.yaml_constructors = self.yaml_constructors.copy()  # Grab an instance copy from the class
        self.yaml_constructors[self.DEFAULT_MAPPING_TAG] = self._make_ordered_dict
        self.yaml_constructors[None] = construct_tagged_object

    @staticmethod
    def _make_ordered_dict(loader: yaml.Loader, node: yaml.Node) -> OrderedDict:
        loader.flatten_mapping(node)
        return OrderedDict(loader.construct_pairs(node))