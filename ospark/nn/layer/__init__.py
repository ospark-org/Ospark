from typing import NoReturn
from ospark.nn.component.basic_module import BasicModule
import tensorflow as tf 

class Layer(BasicModule):

    def __init__(self, obj_name: str) -> NoReturn:
        super().__init__(obj_name=obj_name)

    def initialize(self) -> NoReturn:
        raise NotImplementedError()

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        return self.model(input_data)