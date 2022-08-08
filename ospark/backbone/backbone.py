from ospark.nn.component.basic_module import ModelObject
from typing import NoReturn, Optional
import tensorflow as tf

class Backbone(ModelObject):

    def __init__(self,
                 obj_name: str,
                 use_catch: Optional[bool]=None,
                 trainable: Optional[bool]=None):
        super().__init__(obj_name=obj_name)
        self._trainable = trainable or True
        self._use_catch = use_catch or False
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

    def in_creating(self) -> NoReturn:
        raise NotImplementedError("Is required")

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError("Is required")