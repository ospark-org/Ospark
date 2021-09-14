from ospark.former.former import Former
from ospark.nn.component.basic_module import BasicModule
from ospark.nn.component.normalization import Normalization
from typing import List, NoReturn, Optional
import tensorflow as tf
import ospark


class Transformer(Former):
 
    def __init__(self,
                 obj_name: str,
                 encoder_blocks: List[BasicModule], 
                 class_number: int, 
                 embedding_size: int,
                 use_graph_mode: bool=True,
                 decoder_blocks: Optional[List[BasicModule]]=None,
                 max_length: int=2000,
                 normalization: Optional[Normalization]=None,
                 initial_norm: bool=False) -> NoReturn:
        super().__init__(obj_name=obj_name,
                         encoder_blocks=encoder_blocks,
                         class_number=class_number,
                         embedding_size=embedding_size,
                         use_graph_mode=use_graph_mode,
                         decoder_blocks=decoder_blocks,
                         max_length=max_length
                         )
        self._normalization = normalization or ospark.normalization.LayerNormalization()
        self._initial_norm  = initial_norm

    @property
    def initial_norm(self) -> bool:
        return self._initial_norm

    @property
    def normalization(self) -> Normalization:
        return self._normalization

    def model(self, encoder_input: tf.Tensor, decoder_input: Optional[tf.Tensor]=None) -> tf.Tensor:
        padding_mask, encoder_encodding_mask, prediction_mask = self.create_mask_matrix(encoder_input)
        encoder_input = self.positional_encoding(encoder_input, encoder_encodding_mask)
        if self.initial_norm:
            encoder_input = self.normalization(encoder_input)
        output = encoder_input
        for encoder_block in self.encoder_blocks:
            output = encoder_block(output, padding_mask)
        if self.decoder_blocks != []:
            padding_mask, decoder_encodding_mask, prediction_mask = self.create_mask_matrix(decoder_input)
            decoder_input = self.positional_encoding(decoder_input, decoder_encodding_mask)
            encoder_output = output * encoder_encodding_mask
            output = decoder_input
            for decoder_block in self.decoder_blocks:
                output = decoder_block(input_data=output, encoder_output=encoder_output, mask=padding_mask)
        prediction = self.classifier(tf.matmul(output, self.classify_layer.value))
        return prediction * prediction_mask