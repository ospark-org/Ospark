from ospark.former.former import Former
from ospark.nn.block import Block
from typing import List, NoReturn, Optional
import tensorflow as tf

class Informer(Former):

    def __init__(self,
                 obj_name: str,
                 encoder_blocks: List[Block],
                 class_number: int,
                 embedding_size: int,
                 use_graph_mode: bool=True,
                 decoder_blocks: Optional[List[Block]]=None,
                 max_length: int = 2000,
                 ) -> NoReturn:
        super().__init__(obj_name=obj_name,
                         encoder_blocks=encoder_blocks,
                         class_number=class_number,
                         embedding_size=embedding_size,
                         use_graph_mode=use_graph_mode,
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