from typing import NoReturn, Optional
from ospark.nn.component.basic_module import ModelObject
import tensorflow as tf


class Layer(ModelObject):

    def __init__(self, obj_name: str, is_training: Optional[bool]=None, training_phase: Optional[bool]=None) -> NoReturn:
        super().__init__(obj_name=obj_name, is_training=is_training, training_phase=training_phase)

    def in_creating(self) -> NoReturn:
        pass

    def pipeline(self, *args, **kwargs) -> tf.Tensor:
        raise NotImplementedError()