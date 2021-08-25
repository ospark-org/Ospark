from ospark.nn.block import Block
from typing import NoReturn, Optional
from ospark.nn.layer.convolution_layer import ConvolutionLayer
import tensorflow as tf


class ResnetBlock(Block):

    def __init__(self,
                 obj_name: str,
                 input_channel: int,
                 scale_rate: Optional[int]=4,
                 kernel_size: Optional[int]=3,
                 strides: Optional[int]=1,
                 shortcut_conv: Optional[bool]=False,
                 trainable: Optional[bool]=True) -> NoReturn:
        super().__init__(obj_name=obj_name)
        self._input_channel  = input_channel
        self._scale_rate     = scale_rate
        self._kernel_size    = kernel_size
        self._strides        = strides
        self._shortcut_conv  = shortcut_conv
        self._trainable      = trainable

    @property
    def input_channel(self) -> int:
        return self._input_channel

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
    def shortcut_conv(self) -> bool:
        return self._shortcut_conv

    @property
    def trainable(self) -> bool:
        return self._trainable

    def initialize(self) -> NoReturn:
        if self.shortcut_conv:
            self.assign(component=ConvolutionLayer.bn_relu_conv(obj_name="shortcut_conv",
                                                                filter_size=[1, 1, self.input_channel, self.scale_rate * self.input_channel],
                                                                strides=[1, self.strides, self.strides, 1],
                                                                padding="SAME",
                                                                trainable=self.trainable),
                        name="shortcut_conv")

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        if self.shortcut_conv:
            shortcut = self.assigned.shortcut_conv(input_data)
        else:
            shortcut = input_data
        return shortcut


class Block1(ResnetBlock):

    def initialize(self) -> NoReturn:
        super().initialize()
        if self.shortcut_conv:
            self.assign(component=ConvolutionLayer.bn_relu_conv(obj_name="layer_1X1_0",
                                                                filter_size=[1, 1, self.input_channel, self.input_channel],
                                                                strides=[1, self.strides, self.strides, 1],
                                                                padding="SAME",
                                                                trainable=self.trainable))
        else:
            self.assign(component=ConvolutionLayer.bn_relu_conv(obj_name="layer_1X1",
                                                                filter_size=[1, 1, self.scale_rate * self.input_channel, self.input_channel],
                                                                strides=[1, self.strides, self.strides, 1],
                                                                padding="SAME",
                                                                trainable=self.trainable))

        self.assign(component=ConvolutionLayer.bn_relu_conv(obj_name="layer_3X3",
                                                            filter_size=[self.kernel_size, self.kernel_size, self.input_channel, self.input_channel],
                                                            strides=[1, 1, 1, 1],
                                                            padding="SAME",
                                                            trainable=self.trainable))
        self.assign(component=ConvolutionLayer.bn_relu_conv(obj_name="layer_1X1_1",
                                                            filter_size=[1, 1, self.input_channel, self.scale_rate * self.input_channel],
                                                            strides=[1, self.strides, self.strides, 1],
                                                            padding="SAME",
                                                            trainable=self.trainable))

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        output = input_data
        shortcut = super().model(input_data)
        layers = [self.assigned.layer_1X1_0, self.assigned.layer_3X3, self.assigned.layer_1X1_1]
        for layer in layers:
            output = layer(output)
        return output + shortcut

class Block1(ResnetBlock):

    def initialize(self) -> NoReturn:
        super().initialize()
        if self.shortcut_conv:
            self.assign(component=ConvolutionLayer.bn_relu_conv(obj_name="layer_3X3_0",
                                                                filter_size=[3, 3, self.input_channel, self.input_channel],
                                                                strides=[1, self.strides, self.strides, 1],
                                                                padding="SAME",
                                                                trainable=self.trainable))
        else:
            self.assign(component=ConvolutionLayer.bn_relu_conv(obj_name="layer_3X3_0",
                                                                filter_size=[3, 3, self.scale_rate * self.input_channel, self.input_channel],
                                                                strides=[1, self.strides, self.strides, 1],
                                                                padding="SAME",
                                                                trainable=self.trainable))
        self.assign(component=ConvolutionLayer.bn_relu_conv(obj_name="layer_3X3_1",
                                                            filter_size=[3, 3, self.input_channel, self.scale_rate * self.input_channel],
                                                            strides=[1, 1, 1, 1],
                                                            padding="SAME",
                                                            trainable=self.trainable))

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        output = input_data
        shortcut = super().model(input_data)
        layers = [self.assigned.layer_3X3_0, self.assigned.layer_3X3_1]
        for layer in layers:
            output = layer(output)
        return output + shortcut

