from __future__ import annotations
from ospark.nn.component.basic_module import BasicModule
from typing import List, NoReturn
from ospark.nn.layer import Layer
from abc import abstractmethod
import tensorflow as tf 

class Block(BasicModule):

    def __init__(self, obj_name: str) -> NoReturn:
        super().__init__(obj_name=obj_name)

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        output = input_data
        for layer in self.assigned:
            output = layer(input_data=output)
        return output

    @abstractmethod
    def initialize(self) -> NoReturn:
        pass

    @classmethod
    def draw_structure(cls, layer_classes: List[Layer]):
        def set_configs(obj_name: str, configs: list) -> Block:
            block = cls(obj_name=obj_name)
            for layer, config in zip(layer_classes, configs):
                block.assign(layer(**config))
            return block
        return set_configs

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        return self.model(input_data)