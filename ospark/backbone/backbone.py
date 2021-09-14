from ospark.nn.component.basic_module import BasicModule
from typing import NoReturn, Optional
import tensorflow as tf

class Backbone(BasicModule):

    def __init__(self,
                 obj_name: str,
                 use_catch: Optional[bool]=False,
                 trainable: Optional[bool]=True):
        super().__init__(obj_name=obj_name)
        self._trainable = trainable
        self._use_catch = use_catch
        self._catch_box = []

    @property
    def trainable(self) -> bool:
        return self._trainable

    @property
    def use_catch(self) -> bool:
        return self._use_catch

    @property
    def catch_box(self) -> list:
        return self._catch_box

    def on_creating(self) -> NoReturn:
        return NotImplementedError()

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        return NotImplementedError()

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        return self.model(input_data=input_data)