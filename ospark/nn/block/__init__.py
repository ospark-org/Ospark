from __future__ import annotations
from ospark.nn.component.basic_module import ModelObject
from typing import List, NoReturn, Union, Optional
from ospark.nn.layers import Layer
from abc import abstractmethod
import tensorflow as tf 

class Block(ModelObject):

    def __init__(self, obj_name: str, is_training: Optional[bool]=None) -> NoReturn:
        super().__init__(obj_name=obj_name, is_training=is_training)
        self._layers = []

    @property
    def layers(self) -> List[Layer]:
        return self._layers

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        output = input_data
        for layer in self.assigned:
            output = layer(input_data=output)
        return output

    def in_creating(self) -> NoReturn:
        for layer in self.layers:
            self.assign(component=layer)

    @classmethod
    def draw_structure(cls, layer_classes: List[Layer]):
        def set_configs(obj_name: str, configs: list) -> Block:
            block = cls(obj_name=obj_name)
            for layer, config in zip(layer_classes, configs):
                block.assign(layer(**config))
            return block
        return set_configs

    def add_layers(self, layers: Union[ModelObject, List[ModelObject]]):
        if isinstance(layers, ModelObject):
            self._layers.append(layers)
        elif isinstance(layers, list):
            self._layers += layers
        else:
            raise TypeError("Input type must be ModelObject or List[ModelObject]")