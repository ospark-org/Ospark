from ospark.nn.block import Block
from typing import NoReturn, Optional
from ospark.nn.layers.convolution_layer import ConvolutionLayer
import tensorflow as tf


class ResnetBlock(Block):

    def __init__(self,
                 obj_name: str,
                 input_channel: int,
                 main_channel: int,
                 scale_rate: Optional[int]=4,
                 kernel_size: Optional[int]=3,
                 strides: Optional[int]=1,
                 use_shortcut_conv: Optional[bool]=False,
                 is_training: Optional[bool]=True) -> NoReturn:
        super().__init__(obj_name=obj_name, is_training=is_training)
        self._input_channel  = input_channel
        self._main_channel   = main_channel
        self._scale_rate     = scale_rate
        self._kernel_size    = kernel_size
        self._strides        = strides

        self._use_shortcut_conv  = use_shortcut_conv

    @property
    def input_channel(self) -> int:
        return self._input_channel

    @property
    def main_channel(self) -> int:
        return self._main_channel

    @property
    def scale_rate(self) -> int:
        return self._scale_rate

    @property
    def kernel_size(self) -> int:
        return self._kernel_size

    @property
    def strides(self) -> int:
        return self._strides

    @property
    def use_shortcut_conv(self) -> bool:
        return self._use_shortcut_conv

    def in_creating(self) -> NoReturn:
        if self.use_shortcut_conv:
            self.shortcut_conv = ConvolutionLayer.bn_relu_conv(obj_name="shortcut_conv",
                                                               filter_size=[1,
                                                                            1,
                                                                            self.input_channel,
                                                                            self.scale_rate * self.main_channel],
                                                               strides=[1, self.strides, self.strides, 1],
                                                               padding="SAME",
                                                               is_training=self.is_training)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        if self.use_shortcut_conv:
            shortcut = self.shortcut_conv.pipeline(input_data)
        else:
            shortcut = input_data
        return shortcut


class BottleneckBuildingBlock(ResnetBlock):

    def in_creating(self) -> NoReturn:
        super().in_creating()
        if self.use_shortcut_conv:
            self.layer_1X1_0 = ConvolutionLayer.conv_bn_relu(obj_name="layer_1X1_0",
                                                             filter_size=[1,
                                                                          1,
                                                                          self.input_channel,
                                                                          self.main_channel],
                                                             strides=[1, self.strides, self.strides, 1],
                                                             padding="SAME",
                                                             is_training=self.is_training)
        else:
            self.layer_1X1_0 = ConvolutionLayer.conv_bn_relu(obj_name="layer_1X1_0",
                                                             filter_size=[1,
                                                                          1,
                                                                          self.scale_rate * self.main_channel,
                                                                          self.main_channel],
                                                             strides=[1, self.strides, self.strides, 1],
                                                             padding="SAME",
                                                             is_training=self.is_training)

        self.layer_3X3 = ConvolutionLayer.conv_bn_relu(obj_name="layer_3X3",
                                                       filter_size=[self.kernel_size,
                                                                    self.kernel_size,
                                                                    self.main_channel,
                                                                    self.main_channel],
                                                       strides=[1, 1, 1, 1],
                                                       padding="SAME",
                                                       is_training=self.is_training)
        self.layer_1X1_1 = ConvolutionLayer.conv_bn_relu(obj_name="layer_1X1_1",
                                                         filter_size=[1,
                                                                      1,
                                                                      self.main_channel,
                                                                      self.scale_rate * self.main_channel],
                                                         strides=[1, 1, 1, 1],
                                                         padding="SAME",
                                                         is_training=self.is_training)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        output = input_data
        shortcut = super().pipeline(input_data)
        layers = [self.layer_1X1_0, self.layer_3X3, self.layer_1X1_1]
        for layer in layers:
            output = layer.pipeline(output)
        return output + shortcut

class BuildingBlock(ResnetBlock):

    def in_creating(self) -> NoReturn:
        super().in_creating()
        if self.use_shortcut_conv:
            self.layer_3X3_0 = ConvolutionLayer.conv_bn_relu(obj_name="layer_3X3_0",
                                                             filter_size=[3,
                                                                          3,
                                                                          self.input_channel,
                                                                          self.main_channel],
                                                             strides=[1, self.strides, self.strides, 1],
                                                             padding="SAME",
                                                             is_training=self.is_training)
        else:
            self.layer_3X3_0 = ConvolutionLayer.conv_bn_relu(obj_name="layer_3X3_0",
                                                             filter_size=[3,
                                                                          3,
                                                                          self.scale_rate * self.main_channel,
                                                                          self.main_channel],
                                                             strides=[1, self.strides, self.strides, 1],
                                                             padding="SAME",
                                                             is_training=self.is_training)
        self.layer_3X3_1 = ConvolutionLayer.conv_bn_relu(obj_name="layer_3X3_1",
                                                         filter_size=[3,
                                                                      3,
                                                                      self.main_channel,
                                                                      self.scale_rate * self.main_channel],
                                                         strides=[1, 1, 1, 1],
                                                         padding="SAME",
                                                         is_training=self.is_training)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        output = input_data
        shortcut = super().pipeline(input_data)
        layers = [self.layer_3X3_0, self.layer_3X3_1]
        for layer in layers:
            output = layer.pipeline(output)
        return output + shortcut

