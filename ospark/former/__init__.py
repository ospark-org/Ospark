from ospark.nn.component.basic_module import BasicModule
from abc import abstractmethod
from typing import NoReturn
import tensorflow as tf


class Former(BasicModule):

    def __init__(self, obj_name: str) -> NoReturn:
        super().__init__(obj_name=obj_name)

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def create(self) -> NoReturn:
        super().create("model")
        
    @tf.function
    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        return self.model(input_data)