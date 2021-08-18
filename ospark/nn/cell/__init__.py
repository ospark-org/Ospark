from __future__ import annotations
from ospark.nn.block import Block
from ospark.nn.component.basic_module import BasicModule
from typing import List, NoReturn
from abc import abstractmethod
import tensorflow as tf

class Cell(BasicModule):

    def __init__(self, obj_name: str) -> NoReturn:
        super().__init__(obj_name=obj_name)
        self.blocks  = []

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        output = input_data
        for block in self.assigned:
            output = block(input_data=output)
        return output

    @abstractmethod
    def initialize(self) -> NoReturn:
        pass

    @classmethod
    def draw_structure(cls, block_classes: List[Block]):
        def set_configs(obj_name: str, configs) -> Cell:
            cell = cls(obj_name=obj_name)
            for block, config in zip(block_classes, configs):
                cell.assign(block(**config))
            return cell
        return set_configs

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        return self.model(input_data)