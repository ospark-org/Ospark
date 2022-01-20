from ospark.backbone.backbone import Backbone
from ospark.nn.block import Block
from typing import NoReturn
import tensorflow as tf


class Autoencoder(Backbone):

    def __init__(self,
                 obj_name: str,
                 encoder: Block,
                 decoder: Block):
        super().__init__(obj_name=obj_name)
        self._encoder   = encoder
        self._decoder   = decoder

    @property
    def encoder(self) -> Block:
        return self._encoder

    @property
    def decoder(self) -> Block:
        return self._decoder

    def in_creating(self) -> NoReturn:
        self.assign(component=self.encoder, name="encoder")
        self.assign(component=self.decoder, name="decoder")

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        encoder_output = self.assigned.encoder(input_data)
        decoder_output = self.assigned.decoder(encoder_output)
        return decoder_output