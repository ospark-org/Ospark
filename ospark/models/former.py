import ospark.utility.weight_initializer
from ospark.nn.layers.embedding_layer import EmbeddingLayer
from ospark.nn.block import Block
from ospark.nn.component.weight import Weight
from typing import List, Tuple, Optional, Callable, Union
from ospark import Model
import ospark
import numpy as np
import tensorflow as tf


class Former(Model):

    def __init__(self,
                 obj_name: str,
                 encoder_blocks: List[Block],
                 class_number: int,
                 embedding_size: int,
                 dropout_rate: float,
                 trained_weights: Optional[dict]=None,
                 decoder_blocks: Optional[List[Block]]=None,
                 is_training: Optional[bool]=None,
                 encoder_corpus_size: Optional[int]=None,
                 decoder_corpus_size: Optional[int]=None,
                 use_embedding_layer: Optional[bool]=True,
                 use_classifier: Optional[bool]=False,
                 max_length: int=2000
                 ):
        super().__init__(obj_name=obj_name, trained_weights=trained_weights, is_training=is_training)
        self._embedding_size       = embedding_size
        self._max_length           = max_length
        self._encoder_blocks       = encoder_blocks
        self._decoder_blocks       = decoder_blocks or []
        self._class_number         = class_number
        self._embedding_layer      = None
        self._embedding_scale_rate = tf.math.sqrt(tf.cast(self.embedding_size, dtype=tf.float32))
        self._encoder_dropout_layer= tf.keras.layers.Dropout(rate=dropout_rate)
        self._classify_layer       = ospark.utility.weight_initializer.glorot_uniform(obj_name="classify_layer",
                                                                                      shape=[self.embedding_size, class_number])
        self._classify_layer_bias  = ospark.utility.weight_initializer.zeros(obj_name="classify_layer_bias",
                                                                             shape=[class_number])
        self._classifier           = tf.nn.sigmoid if class_number == 2 else tf.nn.softmax
        self._use_classifier       = use_classifier

        self._positional_encoding_table = self.create_positional_encoding_table()

        for component in [*self.encoder_blocks, self.classify_layer, self.classify_layer_bias]:
            self.assign(component)

        if decoder_blocks is not None:
            self._decoder_dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
            for component in decoder_blocks:
                self.assign(component)

        if use_embedding_layer:
            if encoder_corpus_size is None:
                raise KeyError("Use embedding layers, must setting encoder_corpus_size")
            self._encoder_embedding_layer = EmbeddingLayer(obj_name="encoder_embedding_layer",
                                                           embedding_dimension=embedding_size,
                                                           corpus_size=encoder_corpus_size)
            self.assign(self.encoder_embedding_layer)

            if decoder_blocks is not None:
                if decoder_corpus_size is None:
                    raise KeyError("Use embedding layers, must setting decoder_corpus_size")
                self._decoder_embedding_layer = EmbeddingLayer(obj_name="decoder_embedding_layer",
                                                               embedding_dimension=embedding_size,
                                                               corpus_size=decoder_corpus_size)
                self.assign(self.decoder_embedding_layer)

            self._create_mask_matrix = self.create_mask_by_onehot

        else:
            self._decoder_embedding_layer = lambda input_data: input_data
            self._encoder_embedding_layer = lambda input_data: input_data
            self._create_mask_matrix      = self.create_mask_by_embedding

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def positional_encoding_table(self) -> tf.Tensor:
        return self._positional_encoding_table

    @property
    def encoder_blocks(self) -> List[Block]:
        return self._encoder_blocks

    @property
    def decoder_blocks(self) -> List[Block]:
        return self._decoder_blocks

    @property
    def class_number(self) -> int:
        return self._class_number

    @property
    def classify_layer(self) -> Weight:
        return self._classify_layer

    @property
    def classifier(self) -> Callable[[tf.Tensor], tf.Tensor]:
        return self._classifier

    @property
    def use_classifier(self) -> bool:
        return self._use_classifier

    @property
    def encoder_embedding_layer(self) -> Union[EmbeddingLayer, Callable[[tf.Tensor], tf.Tensor]]:
        return self._encoder_embedding_layer

    @property
    def decoder_embedding_layer(self) -> Union[EmbeddingLayer, Callable[[tf.Tensor], tf.Tensor]]:
        return self._decoder_embedding_layer

    @property
    def embedding_scale_rate(self) -> tf.Tensor:
        return self._embedding_scale_rate

    @property
    def classify_layer_bias(self) -> Weight:
        return self._classify_layer_bias

    @property
    def create_mask_matrix(self) -> Callable[[tf.Tensor, Optional[tf.Tensor]], Tuple[tf.Tensor, tf.Tensor]]:
        return self._create_mask_matrix

    @property
    def encoder_dropout_layer(self) -> tf.keras.layers.Dropout:
        return self._encoder_dropout_layer

    @property
    def decoder_dropout_layer(self) -> tf.keras.layers.Dropout:
        return self._decoder_dropout_layer

    def create_mask_by_onehot(self,
                              encoder_input: tf.Tensor,
                              decoder_input: Optional[tf.Tensor]=None) -> Tuple[tf.Tensor, tf.Tensor]:
        encoder_padding_mask = tf.cast(tf.math.equal(encoder_input, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
        if decoder_input is not None:
            sequence_length        = tf.shape(decoder_input)[1]
            decoder_padding_mask   = tf.cast(tf.math.equal(decoder_input, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
            lookahead_padding_mask = 1 - tf.linalg.band_part(tf.ones((sequence_length, sequence_length)), -1, 0)
            lookahead_padding_mask = tf.maximum(decoder_padding_mask, lookahead_padding_mask)
        else:
            lookahead_padding_mask = None
        return encoder_padding_mask, lookahead_padding_mask

    def create_mask_by_embedding(self,
                                 encoder_input: tf.Tensor,
                                 decoder_input: Optional[tf.Tensor]=None) -> Tuple[tf.Tensor, tf.Tensor]:
        encoder_input = encoder_input[:, :, 0]
        decoder_input = decoder_input[:, :, 0] if decoder_input is not None else None
        encoder_padding_mask, lookahead_padding_mask = self.create_mask_by_onehot(encoder_input=encoder_input,
                                                                                  decoder_input=decoder_input)
        return encoder_padding_mask, lookahead_padding_mask

    def create_positional_encoding_table(self) -> tf.Tensor:
        basic_table = np.zeros(shape=[self.max_length, self.embedding_size])
        position    = np.arange(self.max_length).reshape([-1, 1])
        denominator = np.power(10000, np.arange(0, self.embedding_size, 2) / self.embedding_size)
        basic_table[:, 0::2] = np.sin(position / denominator)
        basic_table[:, 1::2] = np.cos(position / denominator)
        return tf.convert_to_tensor(basic_table, dtype=tf.float32)[tf.newaxis, :, :]

    def pipeline(self, encoder_input: tf.Tensor, decoder_input: Optional[tf.Tensor]=None) -> tf.Tensor:
        raise NotImplementedError()