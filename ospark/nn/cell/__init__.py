from __future__ import annotations
from ospark.nn.block import Block
from ospark.nn.component.basic_module import ModelObject
from typing import List, NoReturn, Optional
from abc import abstractmethod
import tensorflow as tf

class Cell(ModelObject):

    def __init__(self, obj_name: str, is_training: Optional[bool]=None) -> NoReturn:
        super().__init__(obj_name=obj_name, is_training=is_training)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        output = input_data
        for block in self.assigned:
            output = block.pipeline(input_data=output)
        return output

    @abstractmethod
    def in_creating(self) -> NoReturn:
        pass

    @classmethod
    def draw_structure(cls, block_classes: List[Block]):
        def set_configs(obj_name: str, configs) -> Cell:
            cell = cls(obj_name=obj_name)
            for block, config in zip(block_classes, configs):
                cell.assign(block(**config))
            return cell
        return set_configs