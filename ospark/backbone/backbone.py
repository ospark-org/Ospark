from ospark.nn.component.basic_module import ModelObject
from typing import NoReturn, Optional
import tensorflow as tf

class Backbone(ModelObject):

    def __init__(self,
                 obj_name: str,
                 is_keep_blocks: Optional[bool]=None,
                 is_training: Optional[bool]=None):
        super().__init__(obj_name=obj_name, is_training=is_training)
        if is_keep_blocks is not None:
            self._is_keep_blocks = is_keep_blocks
        else:
            self._is_keep_blocks = False
        self._block_register = []

    @property
    def is_keep_blocks(self) -> bool:
        return self._is_keep_blocks

    @property
    def block_register(self) -> list:
        return self._block_register

    def in_creating(self) -> NoReturn:
        raise NotImplementedError("Is required")

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError("Is required")