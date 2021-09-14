from ospark.nn.component.basic_module import BasicModule
from ospark.backbone.backbone import Backbone
from typing import NoReturn
from ospark import Weight
import tensorflow as tf

class Model(BasicModule):

    def __init__(self,
                 obj_name: str,
                 classify_layer: Weight):
        super().__init__(obj_name=obj_name)
        self._classify_layer = classify_layer

    @property
    def classify_layer(self) -> Weight:
        return self._classify_layer

    def on_creating(self) -> NoReturn:
        self.assign(component=self.classify_layer, name="classify_layer")

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        return NotImplementedError()

    @tf.function
    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        output = self.model(input_data=input_data)
        return output