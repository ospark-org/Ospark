from ospark.nn.model import Model
from ospark.backbone.backbone import Backbone
from ospark import Weight
from typing import NoReturn, Optional, List
from ospark.backbone.u_net import shared_convolution
import tensorflow as tf
import ospark
from memory_profiler import profile


class PixelWiseDetection(Model):

    def __init__(self,
                 obj_name: str,
                 backbone: Backbone,
                 classify_layer: Weight,
                 padding: Optional[str]="SAME",
                 strides: Optional[List[int]]=[1, 1, 1, 1]):
        super().__init__(obj_name=obj_name,
                         backbone=backbone,
                         classify_layer=classify_layer)
        self._padding        = padding
        self._strides        = strides

    @property
    def padding(self) -> str:
        return self._padding

    @property
    def strides(self) -> List[int]:
        return self._strides

    def initialize(self) -> NoReturn:
        self.assign(component=self.backbone, name="backbone")
        self.assign(component=self.classify_layer, name="classify_layer")

    @profile
    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        output = self.assigned.backbone(input_data)
        prediction = tf.nn.conv2d(output, self.assigned.classify_layer, strides=self.strides, padding=self.padding)
        return prediction

def fots_detection_model(trainable: bool):
    backbone = shared_convolution(trainable=trainable)
    classify_layer = ospark.weight.truncated_normal(obj_name="classify_layer",
                                                    weight_shape=[1, 1, 32, 6])
    model = PixelWiseDetection(obj_name="detection_model",
                               backbone=backbone,
                               classify_layer=classify_layer)
    return model