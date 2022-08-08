from ospark import Model
from ospark.nn.block import Block
from ospark.nn.layers import Layer
from ospark.nn.layers.dense_layer import DenseLayer
from ospark.nn.block.vgg_block import fots_like_vgg
from ospark.nn.block.transformer_block import transformer_encoder_block
from typing import List, Optional, Callable
from functools import reduce
import tensorflow as tf


class TextRecognition(Model):

    def __init__(self,
                 obj_name: str,
                 classify_layer: Layer,
                 sequential_conv: List[Block],
                 sequential_model: List[Block],
                 trained_weights: Optional[dict]=None):
        super().__init__(obj_name=obj_name, trained_weights=trained_weights)
        self._classify_layer   = classify_layer
        self._sequential_conv  = sequential_conv
        self._sequential_model = sequential_model

    @property
    def classify_layer(self) -> Layer:
        return self._classify_layer

    @property
    def sequential_conv(self) -> List[Block]:
        return self._sequential_conv

    @property
    def sequential_model(self) -> List[Block]:
        return self._sequential_model

    # def in_creating(self) -> NoReturn:
    #     for layer in self.sequential_conv:
    #         self.assign(component=layer)
    #     for layer in self.sequential_model:
    #         self.assign(component=layer)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(tf.math.not_equal(input_data[:, 0, :, 0], 0), dtype=tf.float32)
        sequential_conv_output = reduce(lambda output, layer: layer.pipeline(output), self.sequential_conv, input_data) * mask[:, tf.newaxis, :, tf.newaxis]
        sequence_model_output  = reduce(lambda output, layer: layer.pipeline(output), self.sequential_model, tf.squeeze(sequential_conv_output, axis=1))
        prediction             = self.classify_layer.pipeline(sequence_model_output)
        return prediction


def fots_recognition_model(class_number: int,
                           scale_rate: int,
                           head_number: int,
                           input_channel: int,
                           retrained_weights:dict,
                           sequential_output_channels: List[List[int]],
                           dropout_rate: Optional[float]=0.0,
                           sequential_model_block_number: Optional[int]=4,
                           trainable: Optional[bool]=True
                           ) -> TextRecognition:
    sequential_conv  = fots_like_vgg(input_channel=input_channel,
                                     output_channels=sequential_output_channels,
                                     trainable=trainable)
    sequential_model = []
    for i in range(sequential_model_block_number):
        name  = f"block_{i}"
        block = transformer_encoder_block(obj_name=name,
                                          embedding_size=sequential_output_channels[-1][-1],
                                          head_number=head_number,
                                          scale_rate=scale_rate,
                                          dropout_rate=dropout_rate,
                                          is_training=trainable)
        sequential_model.append(block)

    classify_layer   = DenseLayer(obj_name="classify_layer",
                                  input_dimension=sequential_output_channels[-1][-1],
                                  hidden_dimension=[class_number])
    return TextRecognition(obj_name="recognition_model",
                           trained_weights=retrained_weights,
                           sequential_conv=sequential_conv,
                           sequential_model=sequential_model,
                           classify_layer=classify_layer)