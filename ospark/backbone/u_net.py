from __future__ import annotations
from ospark.backbone.backbone import Backbone
from ospark.backbone.resnet import resnet_50
from ospark.nn.block import Block
from ospark.nn.block.connection_block import shared_convolution_decoder
from typing import NoReturn
import tensorflow as tf
from typing import List


class Unet(Backbone):

    def __init__(self,
                 obj_name: str,
                 down_sampling_net: Backbone,
                 up_sampling_net: Block):
        super().__init__(obj_name=obj_name,
                         use_catch=False,
                         trainable=True)
        self._down_sampling_net = down_sampling_net
        self._up_sampling_net   = up_sampling_net

    @property
    def down_sampling_net(self) -> Backbone:
        return self._down_sampling_net

    @property
    def up_sampling_net(self) -> Block:
        return self._up_sampling_net

    def on_creating(self) -> NoReturn:
        self.assign(component=self.down_sampling_net, name="down_sampling_net")
        self.assign(component=self.up_sampling_net, name="up_sampling_net")

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        encoder_output = self.down_sampling_net(input_data)
        if type(encoder_output) != list:
            raise TypeError("Type is not list, please check use_catch of backbone attribute")
        decoder_output = self.up_sampling_net(encoder_output)
        return decoder_output

    @classmethod
    def build_shared_conv(cls,
                          input_channels: List[int]=None,
                          output_channels: List[int]=None,
                          trainable: bool=True) -> Unet:
        """

        Args:
            input_channels: List[int]
                default value is [2048 + 1024, 128 + 512, 64 + 256].
            output_channels: List[int]
                default value is [128, 64, 32].
            trainable: bool
                default value is True.
        Returns:
            Implemented Unet.
        """

        encoder         = resnet_50(trainable=trainable, catch_output=True)
        input_channels  = input_channels or [2048 + 1024, 128 + 512, 64 + 256]
        output_channels = output_channels or [128, 64, 32]
        decoder = shared_convolution_decoder(input_channels=input_channels,
                                             output_channels=output_channels,
                                             trainable=trainable)
        shared_convolution = cls(obj_name="shared_convolution",
                                 down_sampling_net=encoder,
                                 up_sampling_net=decoder)
        return shared_convolution

