from ospark.nn.block import Block
from ospark.nn.layers import Layer
from typing import List, Optional
import tensorflow as tf


class EmbeddingDecoderBlock(Block):

    def __init__(self,
                 obj_name: str,
                 layers: List[Layer],
                 is_training: Optional[bool],
                 training_phase: Optional[bool]):
        super().__init__(obj_name=obj_name, layers=layers, is_training=is_training, training_phase=training_phase)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()