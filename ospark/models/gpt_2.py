from ospark import Model
from typing import List, Optional
from ospark.nn.block import Block
from ospark.nn.layers.normalization import LayerNormalization
from ospark.nn.layers.dense_layer import DenseLayer
from ospark.nn.layers.embedding_layer import EmbeddingLayer
from functools import reduce
import tensorflow as tf
import numpy as np


class GPTSecondGeneration(Model):
    
    def __init__(self,
                 obj_name: str,
                 blocks: List[Block],
                 class_number: int,
                 embedding_size: int,
                 corpus_size: int,
                 dropout_rate: float,
                 max_length: Optional[int]=None,
                 trained_weights: Optional[dict]=None,
                 is_training: Optional[bool]=None,
                 use_softmax: Optional[bool]=None):
        super(GPTSecondGeneration, self).__init__(obj_name=obj_name,
                                                  trained_weights=trained_weights,
                                                  is_training=is_training)
        self._layer_norm          = LayerNormalization(embedding_size)
        self._blocks              = blocks
        self._max_length          = max_length or 1024
        self._embedding_size      = embedding_size
        self._corpus_size         = corpus_size
        self._use_softmax         = use_softmax if use_softmax is not None else True
        self._embedding_layer     = EmbeddingLayer(embedding_dimension=embedding_size, corpus_size=corpus_size)
        self._dropout_layer       = tf.keras.layers.Dropout(rate=dropout_rate)
        self._classify_layer      = DenseLayer(obj_name="classify_layer",
                                               input_dimension=embedding_size,
                                               hidden_dimension=[class_number],
                                               is_training=is_training)

        self._embedding_scale_rate      = tf.math.sqrt(tf.cast(embedding_size, dtype=tf.float32))
        self._positional_encoding_table = self.create_positional_encoding_table()

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        input_data  = self._embedding_layer.pipeline(input_data=input_data)
        input_data *= self._embedding_scale_rate
        input_data += self._positional_encoding_table[:, input_data.shape[1], ...]

        dropout_output = self._dropout_layer(input_data, training=self.is_training)
        block_output   = reduce(lambda input_data, block: block.pipeline(input_data=input_data), self._blocks, dropout_output)
        norm_output    = self._layer_norm.pipeline(input_data=block_output)
        result         = self._classify_layer.pipeline(input_data=norm_output)
        if self._use_softmax:
            result = tf.nn.softmax(result)
        return result

    def create_positional_encoding_table(self) -> tf.Tensor:
        basic_table = np.zeros(shape=[self._max_length, self._embedding_size])
        position    = np.arange(self._max_length).reshape([-1, 1])
        denominator = np.power(10000, np.arange(0, self._embedding_size, 2) / self._embedding_size)
        basic_table[:, 0::2] = np.sin(position / denominator)
        basic_table[:, 1::2] = np.cos(position / denominator)
        return tf.convert_to_tensor(basic_table, dtype=tf.float32)[tf.newaxis, :, :]


