import ospark
from ospark.nn.layers.convolution_layer import ConvolutionLayer
from ospark.nn.layers import Layer
from typing import NoReturn, Optional
import tensorflow as tf


class ConnectionLayer(Layer):

    def __init__(self,
                 obj_name: str,
                 concatenated_channel: int,
                 output_channel: int,
                 is_training: Optional[bool]=None):
        super().__init__(obj_name=obj_name, is_training=is_training)
        self._concatenated_channel  = concatenated_channel
        self._output_channel        = output_channel
        self._conv_1                = None
        self._conv_2                = None
        self.default_conv()

    @property
    def concatenated_channel(self) -> int:
        return self._concatenated_channel

    @property
    def output_channel(self) -> int:
        return self._output_channel

    @property
    def conv_1(self) -> ConvolutionLayer:
        return self._conv_1

    @property
    def conv_2(self) -> ConvolutionLayer:
        return self._conv_2

    def set_conv_layer(self,
                   conv_1: ConvolutionLayer,
                   conv_2: ConvolutionLayer) -> NoReturn:
        self._conv_1 = conv_1
        self._conv_2 = conv_2

    def default_conv(self) -> NoReturn:
        self._conv_1 = ConvolutionLayer.conv_bn_relu(obj_name="fusion_layer",
                                                     filter_size=[1, 1, self.concatenated_channel, self.output_channel],
                                                     strides=[1, 1, 1, 1],
                                                     padding="SAME",
                                                     is_training=self.is_training)
        self._conv_2 = ConvolutionLayer.conv_bn_relu(obj_name="conv_layer",
                                                     filter_size=[3, 3, self.output_channel, self.output_channel],
                                                     strides=[1, 1, 1, 1],
                                                     padding="SAME",
                                                     is_training=self.is_training)

    def in_creating(self) -> NoReturn:
        self._convolution_1 = self.conv_1
        self._convolution_2 = self.conv_2
        # self.assign(component=self.conv_1)
        # self.assign(component=self.conv_2)

    def pipeline(self, input_data: tf.Tensor, connection_input: tf.Tensor) -> tf.Tensor:
        shape          = tf.shape(connection_input)
        height, width  = shape[1], shape[2]
        input_data     = tf.image.resize(input_data , [height, width])
        concat_input   = tf.concat([input_data, connection_input], axis=-1)
        output         = self._convolution_1.pipeline(concat_input)
        output         = self._convolution_2.pipeline(output)
        return output

