import math
from ospark.backbone.backbone import Backbone
from ospark.nn.layers import Layer
from typing import *
from ospark.backbone.u_net import Unet
from ospark.nn.layers.convolution_layer import ConvolutionLayer
import tensorflow as tf
from ospark import Model


class PixelWiseDetection(Model):

    def __init__(self,
                 obj_name: str,
                 backbone: Backbone,
                 classify_layer: Layer,
                 trained_weights: Optional[dict]=None):
        super().__init__(obj_name=obj_name, trained_weights=trained_weights)
        self._classify_layer = classify_layer
        self._backbone       = backbone

    @property
    def backbone(self) -> Backbone:
        return self._backbone

    @property
    def classify_layer(self) -> Layer:
        return self._classify_layer

    def pipeline(self, input_data: tf.Tensor) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
        feature_map = self.backbone.pipeline(input_data)
        prediction  = self.classify_layer.pipeline(feature_map)

        score_map, bbox_map, angle_map = prediction[:, :, :, 0: 1], prediction[:, :, :, 1: 5], prediction[:, :, :, 5: 6]

        score_map = tf.nn.sigmoid(score_map)
        bbox_map  = tf.nn.sigmoid(bbox_map) * 224
        angle_map = (tf.nn.sigmoid(angle_map) - .5) * math.pi/2
        return (score_map, bbox_map, angle_map), feature_map

    @tf.function
    def __ceil__(self, input_data: tf.Tensor) -> tf.Tensor:
        prediction, feature_map = self.pipeline(input_data=input_data)
        return prediction, feature_map


def fots_detection_model(trainable: bool, retrained_weights):
    backbone = Unet.build_shared_conv(is_training=trainable)
    classify_layer = ConvolutionLayer(obj_name="classify_layer",
                                      filter_size=[1, 1, 32, 6],
                                      strides=[1, 1, 1, 1],
                                      padding="SAME")
    model = PixelWiseDetection(obj_name="detection_model",
                               trained_weights=retrained_weights,
                               backbone=backbone,
                               classify_layer=classify_layer)
    return model