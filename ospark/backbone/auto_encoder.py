from ospark.backbone.backbone import Backbone
from ospark.nn.model import Model
from ospark.nn.block import Block
from typing import NoReturn, List, Tuple, Union, Optional
from functools import reduce
import tensorflow as tf
import numpy as np


class Autoencoder(Backbone):

    def __init__(self,
                 obj_name: str,
                 encoder: Block,
                 decoder: Block,
                 is_training: Optional[bool]=None,
                 training_phase: Optional[bool]=None):
        super().__init__(obj_name=obj_name, is_training=is_training, training_phase=training_phase)
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

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        encoder_output = self.assigned.encoder.pipeline(input_data)
        decoder_output = self.assigned.decoder.pipeline(encoder_output)
        return decoder_output


class PositionalEncodingAutoencoder(Autoencoder):

    def __init__(self,
                 obj_name: str,
                 encoder: List[Block],
                 decoder: List[Block],
                 embedding_size: int,
                 is_training: Optional[bool]=None,
                 training_phase: Optional[bool]=None):
        super().__init__(obj_name=obj_name,
                         encoder=encoder,
                         decoder=decoder,
                         is_training=is_training,
                         training_phase=training_phase)
        self._embedding_size = embedding_size
        self._year_positional_embedding  = self.create_positional_embedding(period_length=3000)
        self._month_positional_embedding = self.create_positional_embedding(period_length=13)
        self._day_positional_embedding   = self.create_positional_embedding(period_length=32)

    def create_positional_embedding(self, period_length: int) -> tf.Tensor:
        basic_table = np.zeros(shape=[period_length, self._embedding_size])
        position    = np.arange(period_length).reshape([-1, 1])
        denominator = np.power(10000, np.arange(0, self._embedding_size, 2) / self._embedding_size)
        basic_table[:, 0::2] = np.sin(position / denominator)
        basic_table[:, 1::2] = np.cos(position / denominator)
        return tf.convert_to_tensor(basic_table, dtype=tf.float32)[tf.newaxis, :, :]

    def positional_encoding(self, year: int, month: int, day: int) -> tf.Tensor:
        year_position  = self._year_positional_embedding[:, year: year+1, ...]
        month_position = self._month_positional_embedding[:, month: month+1, ...]
        day_position   = self._day_positional_embedding[:, day: day+1, ...]
        return year_position + month_position + day_position

    def pipeline(self, input_data: tf.Tensor, batch_timestamps: List[str]) -> Union[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        input_data = tf.convert_to_tensor(input_data)
        positional_embeddings = []
        for timestamps in batch_timestamps:
            year, month, day = timestamps[0].split("-")
            positional_embedding = self.positional_encoding(year=int(year), month=int(month), day=int(day))
            positional_embeddings.append(positional_embedding)

        positional_embeddings = tf.concat(positional_embeddings, axis=0)

        embedding  = self.encoder[0].pipeline(input_data=input_data / 1000)
        embedding += positional_embeddings

        encoder_output = reduce(lambda input_data, layer: layer.pipeline(input_data=input_data), self.encoder[1:], embedding)
        if not self.training_phase:
            return encoder_output
        decoder_output = self.decoder.pipeline(input_data=encoder_output)
        return decoder_output, input_data