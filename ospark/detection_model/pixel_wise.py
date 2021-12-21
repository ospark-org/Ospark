import math
from ospark.nn.model import Model
from ospark.backbone.backbone import Backbone
from ospark import Weight
from typing import NoReturn, Optional, List, Tuple
from ospark.backbone.u_net import Unet
import tensorflow as tf
import ospark


class PixelWiseDetection(Model):

    def __init__(self,
                 obj_name: str,
                 backbone: Backbone,
                 classify_layer: Weight,
                 padding: Optional[str]="SAME",
                 strides: Optional[List[int]]=[1, 1, 1, 1]):
        super().__init__(obj_name=obj_name,
                         classify_layer=classify_layer)
        self._backbone       =backbone
        self._padding        = padding
        self._strides        = strides

    @property
    def padding(self) -> str:
        return self._padding

    @property
    def strides(self) -> List[int]:
        return self._strides

    @property
    def backbone(self) -> Backbone:
        return self._backbone

    def on_creating(self) -> NoReturn:
        super().on_creating()
        self.assign(component=self.backbone, name="backbone")

    def model(self, input_data: tf.Tensor) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
        feature_map = self.assigned.backbone(input_data)
        prediction  = tf.nn.conv2d(feature_map, self.assigned.classify_layer, strides=self.strides, padding=self.padding)

        score_map, bbox_map, angle_map = prediction[:, :, :, 0: 1], prediction[:, :, :, 1: 5], prediction[:, :, :, 5: 6]

        score_map = tf.nn.sigmoid(score_map)
        bbox_map  = tf.nn.sigmoid(bbox_map) * 224
        angle_map = (tf.nn.sigmoid(angle_map) - .5) * math.pi/2
        return (score_map, bbox_map, angle_map), feature_map

    @tf.function
    def __ceil__(self, input_data: tf.Tensor) -> tf.Tensor:
        prediction, feature_map = self.model(input_data=input_data)
        return prediction, feature_map


def fots_detection_model(trainable: bool):
    backbone = Unet.build_shared_conv(trainable=trainable)
    classify_layer = ospark.weight.truncated_normal(obj_name="classify_layer",
                                                    weight_shape=[1, 1, 32, 6])
    model = PixelWiseDetection(obj_name="detection_model",
                               backbone=backbone,
                               classify_layer=classify_layer)
    return model