from ospark.nn.layer.convolution_layer import ConvolutionLayer
from ospark.nn.layer import Layer
from typing import NoReturn
import tensorflow as tf


class ConnectionLayer(Layer):

    def __init__(self,
                 obj_name: str,
                 input_channel: int,
                 output_channel: int,
                 trainable: bool):
        super().__init__(obj_name=obj_name)
        self._input_channel  = input_channel
        self._output_channel = output_channel
        self._trainable      = trainable
        self._conv_1         = None
        self._conv_2         = None
        self.default_conv()

    @property
    def input_channel(self) -> int:
        return self._input_channel

    @property
    def output_channel(self) -> int:
        return self._output_channel

    @property
    def trainable(self) -> bool:
        return self._trainable

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
        self._conv_1 = ConvolutionLayer.conv_bn_relu(obj_name="reduce_channel_layer",
                                                     filter_size=[1, 1, self.input_channel, self.output_channel],
                                                     strides=[1, 1, 1, 1],
                                                     padding="SAME",
                                                     trainable=self.trainable)
        self._conv_2 = ConvolutionLayer.conv_bn_relu(obj_name="mix_layer",
                                                     filter_size=[3, 3, self.output_channel, self.output_channel],
                                                     strides=[1, 1, 1, 1],
                                                     padding="SAME",
                                                     trainable=self.trainable)

    def on_creating(self) -> NoReturn:
        self.assign(component=self.conv_1)
        self.assign(component=self.conv_2)

    def model(self, input_data: tf.Tensor, connection_input: tf.Tensor) -> tf.Tensor:
        shape          = tf.shape(connection_input)
        height, width  = shape[1], shape[2]
        input_data     = tf.image.resize(input_data , [height, width])
        concat_input   = tf.concat([input_data, connection_input], axis=-1)
        output         = self.assigned.reduce_channel_layer(concat_input)
        output         = self.assigned.mix_layer(output)
        return output

    def __call__(self, input_data: tf.Tensor, connection_input: tf.Tensor) -> tf.Tensor:
        output = self.model(input_data=input_data, connection_input=connection_input)
        return output

