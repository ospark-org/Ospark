from ospark import Model
from typing import List, Optional, Tuple
from ospark.nn.block.transformer_block import transformer_encoder_block
from ospark.nn.layers.dense_layer import DenseLayer
from ospark import weight_initializer
from ospark.algorithm.blokcwise_making import BlockwiseMasking
from functools import reduce
import tensorflow as tf
import numpy as np


class VisionTransformer(Model):

    def __init__(self,
                 obj_name: str,
                 image_height: int,
                 image_width: int,
                 patch_height: int,
                 patch_width: int,
                 head_number: int,
                 encoder_number: int,
                 scale_rate: int,
                 dropout_rate: float,
                 delay_create: Optional[bool]=None,
                 classification_number: Optional[int]=None,
                 embedding_size: Optional[int]=None,
                 is_training: Optional[bool]=None,
                 trained_weights: Optional[dict]=None):
        super().__init__(obj_name=obj_name,
                         delay_create=delay_create,
                         is_training=is_training,
                         trained_weights=trained_weights)
        self._image_height     = image_height
        self._image_width      = image_width
        self._patch_height     = patch_height
        self._patch_width      = patch_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        self._encoder_number   = encoder_number
        self._head_number      = head_number
        self._scale_rate       = scale_rate
        self._input_dimension  = patch_height * patch_width * 3
        self._embedding_size   = embedding_size or self._input_dimension
        self._dropout_rate     = dropout_rate
        self._positional_table = self.create_positional_encoding_table(max_token=int((image_height / patch_height) * (image_width / patch_width)))

        self._classification_number   = classification_number
        self._linear_projection_layer = DenseLayer(obj_name="linear_projection",
                                                   input_dimension=self._input_dimension,
                                                   hidden_dimension=[self._embedding_size],
                                                   is_training=is_training)

        self._output_layer = DenseLayer(obj_name="output_layer",
                                        input_dimension=self._embedding_size,
                                        hidden_dimension=[self._embedding_size * scale_rate, self._embedding_size],
                                        is_training=is_training)

        if classification_number is not None:
            self._classify_layer = DenseLayer(obj_name="classify_layer",
                                              input_dimension=self._embedding_size,
                                              hidden_dimension=[classification_number],
                                              is_training=is_training)

        self._cls = weight_initializer.glorot_uniform(obj_name="cls_token",
                                                      shape=[1, 1, self._embedding_size],
                                                      trainable=is_training)

        self._blocks = []
        for i in range(encoder_number):
            name = f"block_{i}"
            encoder_block = transformer_encoder_block(obj_name=name,
                                                      embedding_size=self._embedding_size,
                                                      head_number=self._head_number,
                                                      scale_rate=self._scale_rate,
                                                      dropout_rate=self._dropout_rate,
                                                      is_training=is_training)
            self._blocks.append(encoder_block)

    def pipeline(self, input_data: tf.Tensor, mask_matrix: Optional[tf.Tensor]=None) -> tf.Tensor:
        shape = tf.shape(input_data)
        batch_size, length = shape[0], shape[1]

        input_sequence  = self.process_image(input_data=input_data, batch_size=batch_size)
        input_sequence  = self._linear_projection_layer.pipeline(input_data=input_sequence) # [B, L, D]
        if mask_matrix is not None:
            input_sequence *= mask_matrix
        input_sequence += self._positional_table
        added_cls       = tf.concat([tf.tile(self._cls, [batch_size, 1, 1]), input_sequence], axis=1)

        encoder_output     = reduce(lambda input_data, block: block.pipeline(input_data=input_data),
                                    self._blocks,
                                    added_cls)

        cls_output    = encoder_output[:, :1, :]
        result        = self._output_layer.pipeline(input_data=cls_output)

        if self._classification_number is not None:
            result = self._classify_layer.pipeline(input_data=result)
        result = tf.concat([result, encoder_output[:, 1:, :]], axis=1)
        return result

    def process_image(self, input_data: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
        # input_data: [B, H, W, C] -> [B, PN, PN, P, P, C] -> [B, L, D]
        # L = PN * PN, D = P * P * C
        chunking_width  = tf.split(input_data, int(self._image_height / self._patch_height), axis=1)
        chunking_height = reduce(lambda init, chunk: init + tf.split(chunk,
                                                                     int(self._image_width / self._patch_width),
                                                                     axis=2),
                                 chunking_width,
                                 [])

        sequence = self.flatten(input_data=chunking_height, batch_size=batch_size)
        return sequence

    def flatten(self, input_data: List[tf.Tensor], batch_size: tf.Tensor) -> tf.Tensor:
        result = tf.concat([tf.reshape(patch, [batch_size, 1, -1]) for patch in input_data], axis=1)
        return result

    def create_positional_encoding_table(self, max_token: int) -> tf.Tensor:
        basic_table = np.zeros(shape=[max_token, self._embedding_size])
        position    = np.arange(max_token).reshape([-1, 1])
        denominator = np.power(10000, np.arange(0, self._embedding_size, 2) / self._embedding_size)
        basic_table[:, 0::2] = np.sin(position / denominator)
        basic_table[:, 1::2] = np.cos(position / denominator)
        return tf.convert_to_tensor(basic_table, dtype=tf.float32)[tf.newaxis, :, :]
