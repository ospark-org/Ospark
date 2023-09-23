from ospark.models.former import Former
from ospark.nn.block import Block
from typing import List, NoReturn, Optional
import tensorflow as tf
import numpy as np

class PositionalEncodingDelegate:

    def __init__(self):
        pass

class TimePositionalEncoding(PositionalEncodingDelegate):

    def __init__(self):
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


class Informer(Former):

    def __init__(self,
                 obj_name: str,
                 encoder_blocks: List[Block],
                 class_number: int,
                 embedding_size: int,
                 dropout_rate: float,
                 positional_encoding_delegate: PositionalEncodingDelegate,
                 use_classify_layer: Optional[bool]=None,
                 trained_weights: Optional[dict]=None,
                 is_training: Optional[bool]=None,
                 decoder_blocks: Optional[List[Block]]=None,
                 max_length: int = 2000,
                 ) -> NoReturn:
        super().__init__(obj_name=obj_name,
                         trained_weights=trained_weights,
                         encoder_blocks=encoder_blocks,
                         class_number=class_number,
                         embedding_size=embedding_size,
                         decoder_blocks=decoder_blocks,
                         max_length=max_length,
                         dropout_rate=dropout_rate,
                         is_training=is_training,
                         use_classify_layer=use_classify_layer)
        self._positional_encoding = positional_encoding_delegate or PositionalEncodingDelegate()

    @property
    def positional_encoding(self):
        return self._positional_encoding

    def pipeline(self, encoder_input: tf.Tensor, decoder_input: tf.Tensor=None) -> tf.Tensor:
        encoder_padding_mask, encoder_encoding_mask, prediction_mask = self.create_mask_matrix(encoder_input=encoder_input)
        encoder_input = self.positional_encoding(encoder_input, encoder_encoding_mask)
        encoder_input = self.encoder_dropout_layer(encoder_input, training=self.is_training)
        output = encoder_input
        for encoder_block in self.encoder_blocks:
            output = encoder_block(output)
        if self.decoder_blocks != []:
            decoder_padding_mask, decoder_encoding_mask, prediction_mask = self.create_mask_matrix(decoder_input)
            decoder_input = self.positional_encoding(decoder_input, decoder_encoding_mask)
            decoder_input = self.decoder_dropout_layer(decoder_input, training=self.is_training)
            encoder_output = output
            output = decoder_input
            for decoder_block in self.decoder_blocks:
                output = decoder_block(input_data=output, encoder_output=encoder_output)
        prediction = self.classifier(tf.matmul(output, self.classify_layer.value))
        return prediction * prediction_mask
