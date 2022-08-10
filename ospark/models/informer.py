from ospark.models.former import Former
from ospark.nn.block import Block
from typing import List, NoReturn, Optional
import tensorflow as tf

class PositionalEncodingDelegate:

    def __init__(self):
        pass

class Informer(Former):

    def __init__(self,
                 obj_name: str,
                 encoder_blocks: List[Block],
                 class_number: int,
                 embedding_size: int,
                 dropout_rate: float,
                 positional_encoding_delegate: PositionalEncodingDelegate,
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
                         is_training=is_training)
        self._positional_encoding = positional_encoding_delegate or PositionalEncodingDelegate()

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
