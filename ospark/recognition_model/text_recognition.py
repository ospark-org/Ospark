from ospark.nn.model import Model
from ospark.nn.block import Block
from ospark.nn.layer import Layer
from ospark.nn.layer.dense_layer import DenseLayer
from ospark.nn.block.vgg_block import fots_like_vgg
from ospark.nn.component.activation import Activation, ReLU
from ospark.nn.component.normalization import Normalization, BatchNormalization
from ospark.nn.block.transformer_block import transformer_encoder_block, create_coder_blocks
from typing import List, NoReturn, Optional, Callable
from functools import reduce
import tensorflow as tf


class TextRecognition(Model):

    def __init__(self,
                 obj_name: str,
                 classify_layer: Layer,
                 sequential_conv: List[Block],
                 sequential_model: List[Block]):
        super().__init__(obj_name=obj_name,
                         classify_layer=classify_layer)
        self._sequential_conv  = sequential_conv
        self._sequential_model = sequential_model

    @property
    def classify_layer(self) -> int:
        return self._classify_layer

    @property
    def sequential_conv(self) -> List[Block]:
        return self._sequential_conv

    @property
    def sequential_model(self) -> List[Block]:
        return self._sequential_model

    def on_creating(self) -> NoReturn:
        super().on_creating()
        for layer in self.sequential_conv:
            self.assign(component=layer)
        for layer in self.sequential_model:
            self.assign(component=layer)

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        sequential_conv_output = reduce(lambda output, layer: layer(output), self.sequential_conv, input_data)
        sequence_model_output  = reduce(lambda output, layer: layer(output), self.sequential_model, tf.squeeze(sequential_conv_output, axis=1))
        prediction             = self.assigned.classify_layer(sequence_model_output)
        return prediction

def fots_recognition_model(class_number: int,
                           scale_rate: int,
                           head_number: int,
                           input_channel: int,
                           sequential_output_channels: List[List[int]],
                           sequential_model_block_number: Optional[int]=4,
                           sequential_model_create_function: Optional[Callable[[str, int, int, int], Block]]=transformer_encoder_block,
                           trainable: Optional[bool]=True
                           ) -> TextRecognition:
    sequential_conv  = fots_like_vgg(input_channel=input_channel,
                                     output_channels=sequential_output_channels,
                                     trainable=trainable)
    sequential_model = create_coder_blocks(block_number=sequential_model_block_number,
                                           create_func=sequential_model_create_function,
                                           embedding_size=sequential_output_channels[-1][-1],
                                           head_number=head_number,
                                           scale_rate=scale_rate)
    classify_layer   = DenseLayer(obj_name="classify_layer",
                                  input_dimension=sequential_output_channels[-1][-1],
                                  hidden_dimension=[class_number])
    return TextRecognition(obj_name="recognition_model",
                           sequential_conv=sequential_conv,
                           sequential_model=sequential_model,
                           classify_layer=classify_layer)