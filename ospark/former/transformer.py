from . import Former
from ospark.nn.component.basic_module import BasicModule
from ospark.nn.component.normalization import Normalization
from ospark.nn.block.transformer_block import transformer
from ospark.nn.component.weight import Weight
from typing import List, Tuple
from enum import Enum
import tensorflow as tf 
import numpy as np
import ospark

class FormerOption(Enum):
    
    transformer = lambda model_name, encoder_number, embedding_size, scale_rate, head_number, class_number: \
                   set_encoder(transformer)(model_name, encoder_number, embedding_size, scale_rate, head_number, class_number)

def set_encoder(encoder):
    def create_model(model_name, encoder_number, embedding_size, scale_rate, head_number, class_number): # TODO
        encoder_blocks = []
        for i in range(encoder_number):
            obj_name = f"block_{i}"
            encoder_blocks.append(encoder(obj_name=obj_name, embedding_size=embedding_size, scale_rate=scale_rate, head_number=head_number))
        return Transformer(model_name, encoder_blocks, class_number, embedding_size)
    return create_model

class Transformer(Former):
 
    def __init__(self, 
                 obj_name: str,
                 encoder_blocks: List[BasicModule], 
                 class_number: int, 
                 embedding_size: int,
                 decoder_blocks: List[BasicModule]=None,
                 max_length: int=2000,
                 normalization: Normalization=None, 
                 initial_norm: bool=True):
        super().__init__(obj_name=obj_name)
        self._embedding_size        = embedding_size
        self._max_length            = max_length
        self._initial_norm          = initial_norm
        self._encoding_table        = self.create_encoding_table()
        self._encoder_blocks        = encoder_blocks
        self._decoder_blocks        = decoder_blocks
        self._class_number          = class_number
        self._normalization         = normalization or ospark.normalization.LayerNormalization()
        self._classify_layer        = ospark.weight.truncated_normal(obj_name="classify_layer",
                                                                     weight_shape=[self.embedding_size, class_number])
        self._classifier            = tf.nn.sigmoid if class_number == 2 else tf.nn.softmax
        for component in [*self.encoder_blocks, self.classify_layer]:
            self.assign(component)
        if decoder_blocks is not None:
            for component in decoder_blocks:
                self.assign(component)

    @property
    def normalization(self) -> Normalization:
        return self._normalization

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def initial_norm(self) -> bool:
        return self._initial_norm

    @property
    def encoding_table(self) -> tf.Tensor:
        return self._encoding_table

    @property
    def encoder_blocks(self) -> List[BasicModule]:
        return self._encoder_blocks

    @property
    def decoder_blocks(self) -> List[BasicModule]:
        return self._decoder_blocks

    @property
    def class_number(self) -> int:
        return self._class_number

    @property
    def classify_layer(self) -> Weight:
        return self._classify_layer

    @property
    def normalization(self) -> Normalization:
        return self._normalization

    @property
    def classifier(self):
        return self._classifier

    @property
    def auto_graph(self) -> bool:
        return self._auto_graph

    def create_mask_matrix(self, input_data: tf.Tensor) -> Tuple[tf.Tensor]:
        padding_mask = tf.cast(tf.math.equal(input_data[:,:,0], 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
        encodding_mask = tf.cast(tf.math.not_equal(input_data[:,:,:], 0), tf.float32)
        prediction_mask = tf.tile(tf.cast(tf.math.not_equal(input_data[:,:,:1], 0), tf.float32), [1, 1, self.class_number])
        return padding_mask, encodding_mask, prediction_mask

    def create_encoding_table(self)-> tf.Tensor:
        basic_table = np.zeros(shape=[self.max_length, self.embedding_size])
        position = np.arange(self.max_length).reshape([-1, 1])
        denominator = np.power(10000, np.arange(0, self.embedding_size, 2) / self.embedding_size)
        basic_table[:, 0::2] = np.sin(position / denominator)
        basic_table[:, 1::2] = np.cos(position / denominator)
        return tf.convert_to_tensor(basic_table, dtype=tf.float32)

    def positional_encoding(self, input_data: tf.Tensor, encodding_mask: tf.Tensor) -> tf.Tensor:
        shape = input_data.shape
        loolkup_index = tf.tile(tf.range(0, shape[1])[tf.newaxis, :], [shape[0], 1])
        input_data += tf.nn.embedding_lookup(self.encoding_table, loolkup_index) * encodding_mask
        return input_data

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        padding_mask, encodding_mask, prediction_mask = self.create_mask_matrix(input_data)
        input_data = self.positional_encoding(input_data, encodding_mask)
        if self.initial_norm:
            input_data = self.normalization(input_data)
        for encoder_block in self.encoder_blocks:
            output = encoder_block(input_data, padding_mask)
            input_data = output
        if self.decoder_blocks is not None:
            for decoder_block in self.decoder_blocks:
                output = decoder_block(input_data)
                input_data = output
        prediction = self.classifier(tf.matmul(output, self.classify_layer.value))
        return prediction * prediction_mask