from __future__ import annotations
from ospark.nn.block import Block
from ospark.nn.layers import Layer
from ospark.nn.layers.normalization import Normalization, BatchNormalization
from ospark.nn.component.activation import Activation, ReLU
from ospark.nn.layers.convolution_layer import ConvolutionLayer
from typing import List, NoReturn, Optional
from functools import reduce
import tensorflow as tf


class VGGBlock(Block):

    def __init__(self,
                 obj_name: str,
                 layers: List[Layer],
                 pooling_size: List[int],
                 pooling_strides: List[int]):
        super().__init__(obj_name=obj_name)
        self._layers          = layers
        self._pooling_size    = pooling_size
        self._pooling_strides = pooling_strides

    @property
    def layers(self) -> List[Layer]:
        return self._layers

    @property
    def pooling_size(self) -> List[int]:
        return self._pooling_size

    @property
    def pooling_sreides(self) -> List[int]:
        return self._pooling_strides

    @classmethod
    def create_vgg_block(cls,
                         obj_name: str,
                         input_channel: int,
                         output_channels: List[int],
                         pooling_size: List[int],
                         pooling_strides: List[int],
                         activation: List[Activation],
                         normalization: List[Normalization],
                         filter_size: Optional[List[int]]=[3, 3],
                         filter_strides: Optional[List[int]]=[1, 1, 1, 1],
                         layer_order: Optional[List[str]]=["conv", "norm", "activate"]) -> VGGBlock:
        layers = []
        for i, output_channel in enumerate(output_channels):
            if i == 0:
                input_channel = input_channel
            else:
                input_channel = output_channels[i - 1]

            layer_name = f"layer_{i}"
            conv_layer = ConvolutionLayer(obj_name=layer_name,
                                          filter_size=[*filter_size, input_channel, output_channel],
                                          strides=filter_strides,
                                          padding="SAME",
                                          activation=activation[i],
                                          normalization=normalization[i],
                                          layer_order=layer_order)
            layers.append(conv_layer)
        return cls(obj_name=obj_name,
                   layers=layers,
                   pooling_size=pooling_size,
                   pooling_strides=pooling_strides)


    def in_creating(self) -> NoReturn:
        for layer in self.layers:
            self.assign(component=layer)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        layer_output = reduce(lambda output, layer: layer.pipeline(output), self.layers, input_data)
        block_output = tf.nn.max_pool2d(input=layer_output,
                                        strides=self.pooling_sreides,
                                        ksize=self.pooling_size,
                                        padding="SAME")
        return block_output

def fots_like_vgg(input_channel: int,
                  output_channels: List[List[int]],
                  trainable: Optional[bool]=True
                  ) -> List[VGGBlock]:
    """
    The vgg model used by FOTS
    
    Args:
        input_channel: int
            Input channel of vgg model, type is int32
        output_channels:  List[List[int]]
            2-D structure [layer_number, output_channels],
    Returns:

    """

    vgg_blocks = []
    for i, layer_channels in enumerate(output_channels):
        if i == 0:
            input_channel = input_channel
        else:
            input_channel = output_channels[i - 1][-1]
        name      = f"vgg_block_{i}"
        vgg_block = VGGBlock.create_vgg_block(obj_name=name,
                                              input_channel=input_channel,
                                              output_channels=layer_channels,
                                              pooling_size=[1, 2, 1, 1],
                                              pooling_strides=[1, 2, 1, 1],
                                              normalization=[BatchNormalization(obj_name="batch_norm",
                                                                                input_depth=channel,
                                                                                is_training=trainable) for channel in layer_channels],
                                              activation=[ReLU() for i in layer_channels])
        vgg_blocks.append(vgg_block)
    return vgg_blocks



