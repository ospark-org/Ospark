from ospark.former.former import Former
from ospark.nn.block import Block
from ospark.nn.block.informer_block import encoder_block, decoder_block
from ospark.nn.component.normalization import Normalization
from typing import List, NoReturn
import tensorflow as tf

class Informer(Former):

    def __init__(self,
                 obj_name: str,
                 encoder_blocks: List[Block],
                 class_number: int,
                 embedding_size: int,
                 decoder_blocks: List[Block] = [],
                 max_length: int = 2000,
                 normalization: Normalization = None
                 ) -> NoReturn:
        super().__init__(obj_name=obj_name,
                         encoder_blocks=encoder_blocks,
                         class_number=class_number,
                         embedding_size=embedding_size,
                         decoder_blocks=decoder_blocks,
                         max_length=max_length)

    def model(self, encoder_input: tf.Tensor, decoder_input: tf.Tensor=None) -> tf.Tensor:
        enocder_padding_mask, encoder_encodding_mask, prediction_mask = self.create_mask_matrix(encoder_input)
        encoder_input = self.positional_encoding(encoder_input, encoder_encodding_mask)
        output = encoder_input
        for encoder_block in self.encoder_blocks:
            output = encoder_block(output)
        if self.decoder_blocks != []:
            decoder_padding_mask, decoder_encodding_mask, prediction_mask = self.create_mask_matrix(decoder_input)
            decoder_input = self.positional_encoding(decoder_input, decoder_encodding_mask)
            encoder_output = output
            output = decoder_input
            for decoder_block in self.decoder_blocks:
                output = decoder_block(input_data=output, encoder_output=encoder_output)
        prediction = self.classifier(tf.matmul(output, self.classify_layer.value))
        return prediction * prediction_mask

    @classmethod
    def quick_build(cls,
                    class_number: int,
                    block_number: int,
                    embedding_size: int,
                    head_number: int,
                    sample_factor: float,
                    filter_width: int=None,
                    pooling_size: list=None,
                    strides: list=None):
        encoder_blocks = []
        decoder_blocks = []
        for i in range(block_number):
            encoder_name = f"encoder_block_{i}"
            encoder_blocks.append(encoder_block(encoder_name, embedding_size, head_number, sample_factor, filter_width, pooling_size, strides))
            decoder_name = f"decoder_block_{i}"
            decoder_blocks.append(decoder_block(decoder_name, embedding_size, head_number, sample_factor))
        return cls("Informer", encoder_blocks, class_number, embedding_size, decoder_blocks)

    def __call__(self, encoder_input: tf.Tensor, decoder_input: tf.Tensor=None) -> tf.Tensor:
        return self.model(encoder_input, decoder_input)