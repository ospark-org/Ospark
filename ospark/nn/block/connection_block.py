from ospark.nn.block import Block
from ospark.nn.layer import Layer
from ospark.nn.layer.connection_layer import ConnectionLayer
from typing import List, NoReturn
from functools import reduce
import tensorflow as tf


class ConnectionBlock(Block):

    def __init__(self,
                 obj_name: str,
                 connection_layers: List[ConnectionLayer]):
        super().__init__(obj_name=obj_name)
        self._connection_layers = connection_layers

    @property
    def connection_layers(self) -> List[Layer]:
        return self._connection_layers

    def on_creating(self) -> NoReturn:
        for layer in self.connection_layers:
            self.assign(component=layer)

    def model(self, input_data: tf.Tensor, connection_input: List[tf.Tensor]) -> tf.Tensor:
        """

        : input_data is 4-D Tensor [b, h, w, c]:
        :param connection_input:
        :return:
        """

        return reduce(
            lambda output, layer: layer(output, connection_input.pop()),
            self.connection_layers,
            input_data)

    def __call__(self, input_data: List[tf.Tensor]) -> tf.Tensor:
        encoder_output = input_data.pop()
        connection_input = input_data
        output = self.model(input_data=encoder_output, connection_input=connection_input)
        return output


def shared_convolution_decoder(input_channels: List[int], output_channels: List[int], trainable: bool) -> ConnectionBlock:
    connection_layers = []
    for i, channels in enumerate(zip(input_channels, output_channels)):
        input_channel, output_channel = channels
        name = f"connection_layer_{i}"
        connection_layer = ConnectionLayer(obj_name=name,
                                           concatenated_channel=input_channel,
                                           output_channel=output_channel,
                                           trainable=trainable)
        connection_layers.append(connection_layer)
    decoder = ConnectionBlock(obj_name="shared_decoder", connection_layers=connection_layers)
    return decoder


