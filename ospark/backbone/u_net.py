from __future__ import annotations
from ospark.nn.block.connection_block import shared_convolution_decoder
from typing import NoReturn, Union, Optional
from ospark.nn.block import Block
from .backbone import Backbone
from .builder import resnet50
import tensorflow as tf
from typing import List


class Unet(Backbone):

    def __init__(self,
                 obj_name: str,
                 downsampling: Union[Block, Backbone],
                 upsampling: Block):
        super().__init__(obj_name=obj_name,
                         use_catch=False,
                         trainable=True)
        self._downsampling = downsampling
        self._upsampling   = upsampling

    @property
    def downsampling(self) -> Backbone:
        return self._downsampling

    @property
    def upsampling(self) -> Block:
        return self._upsampling

    def in_creating(self) -> NoReturn:
        self.assign(component=self.downsampling, name="downsampling")
        self.assign(component=self.upsampling, name="upsampling")

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        encoder_output = self.downsampling(input_data)
        if type(encoder_output) is not list:
            raise TypeError("Type is not list, please check use_catch of backbone attribute")
        decoder_output = self.upsampling(encoder_output)
        return decoder_output

    @classmethod
    def build_shared_conv(cls,
                          input_channels: Optional[List[int]]=None,
                          output_channels: Optional[List[int]]=None,
                          trainable: Optional[bool]=None) -> Unet:
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

        trainable       = trainable or True
        encoder         = resnet50(trainable=trainable, catch_output=True)
        input_channels  = input_channels or [2048 + 1024, 128 + 512, 64 + 256]
        output_channels = output_channels or [128, 64, 32]
        decoder = shared_convolution_decoder(input_channels=input_channels,
                                             output_channels=output_channels,
                                             trainable=trainable)
        shared_convolution = cls(obj_name="shared_convolution",
                                 downsampling=encoder,
                                 upsampling=decoder)
        return shared_convolution

