from ospark.nn.component.basic_module import BasicModule
from typing import NoReturn, Optional
import tensorflow as tf

class Backbone(BasicModule):

    def __init__(self,
                 obj_name: str,
                 catch_output: Optional[bool]=False,
                 trainable: Optional[bool]=True
                 ):
        super().__init__(obj_name=obj_name)
        self._trainable          = trainable
        self._catch_output       = catch_output
        self._catch_box          = []

    @property
    def trainable(self) -> bool:
        return self._trainable

    @property
    def catch_output(self) -> bool:
        return self._catch_output

    @property
    def catch_box(self) -> list:
        return self._catch_box

    def initialize(self) -> NoReturn:
        return NotImplementedError()

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        return NotImplementedError()

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        return self.model(input_data=input_data)