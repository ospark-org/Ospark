from __future__ import annotations
from ospark.nn.block import Block
from ospark.nn.layers.self_attention import SelfAttentionLayer
from ospark.nn.layers.feed_forward import FeedForwardLayer
from typing import Optional, Type
from ospark.nn.layers.normalization import LayerNormalization
import tensorflow as tf


class GPTSecondGenerationBlock(Block):
    """
    GPT-2 architecture block.
    """

    def __init__(self,
                 obj_name: str,
                 attention: SelfAttentionLayer,
                 feedforward: FeedForwardLayer):
        """
        Args:
            obj_name: str
            attention: SelfAttentionLayer
            feedforward: FeedForwardLayer
        """

        super().__init__(obj_name)
        self._attention   = attention
        self._feedforward = feedforward

    @property
    def attention(self) -> SelfAttentionLayer:
        return self._attention

    @property
    def feedforward(self) -> FeedForwardLayer:
        return self._feedforward

    @classmethod
    def create_via_class(cls,
                         obj_name: str,
                         embedding_size: int,
                         head_number: int,
                         scale_rate: int,
                         attention_cls: Optional[Type[SelfAttentionLayer]]=None,
                         feedforward_cls: Optional[Type[FeedForwardLayer]]=None,
                         dropout_rate: Optional[float]=None,
                         is_training: Optional[bool]=None) -> GPTSecondGenerationBlock:
        """
        Make block by parameters and custom layer.

        Args:
            obj_name: str
            embedding_size: int
            head_number: int
            scale_rate: int
            attention_cls: Optional[Type[SelfAttentionLayer]]
            feedforward_cls:Optional[Type[FeedForwardLayer]]
            dropout_rate:Optional[float]
            is_training:Optional[bool]

        Returns:
            object: GPTSecondGenerationBlock
        """

        attention_cls   = attention_cls or SelfAttentionLayer
        feedforward_cls = feedforward_cls or FeedForwardLayer
        dropout_rate    = dropout_rate or 0.0
        return cls(obj_name=obj_name,
                   attention=attention_cls(obj_name="attention",
                                           embedding_size=embedding_size,
                                           head_number=head_number,
                                           use_look_ahead=True,
                                           dropout_rate=dropout_rate,
                                           is_training=is_training),
                   feedforward=feedforward_cls(obj_name="feedforward",
                                               embedding_size=embedding_size,
                                               scale_rate=scale_rate,
                                               dropout_rate=dropout_rate,
                                               is_training=is_training))

    def pipeline(self,
                 input_data: tf.Tensor,
                 padding_mask: tf.Tensor) -> tf.Tensor:
        """
        Pipeline.

        Args:
            input_data: tf.Tensor
            padding_mask: tf.Tensor

        Returns:
            output: tf.Tensor
        """

        output = self.attention.pipeline(input_data=input_data, mask=padding_mask)
        output = self.feedforward.pipeline(input_data=output)
        return output