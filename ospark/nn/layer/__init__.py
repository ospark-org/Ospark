from typing import NoReturn, Optional
from ospark.nn.component.basic_module import BasicModule
import tensorflow as tf 

class Layer(BasicModule):

    def __init__(self, obj_name: str, is_training: Optional[bool]=False) -> NoReturn:
        super().__init__(obj_name=obj_name)
        self._is_training = is_training

    @property
    def is_training(self) -> bool:
        return self._is_training

    def on_creating(self) -> NoReturn:
        raise NotImplementedError()

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        return self.model(input_data)