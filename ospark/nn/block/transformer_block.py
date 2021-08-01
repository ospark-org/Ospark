from __future__ import annotations
from typing import List, NoReturn
from ospark.nn.component.normalization import Normalization
from ospark.nn.layer.self_attention import SelfAttention
from ospark.nn.layer.feed_forward import FeedForward
from ospark.nn.component.activation import Activation
from ospark.nn.component.weight import Weight
from . import Block
import tensorflow as tf
import ospark


class TransformerEncoderBlock(Block):

    def __init__(self, 
                 obj_name: str, 
                 attention: SelfAttention,
                 feedforward: FeedForward) -> NoReturn:
        super().__init__(obj_name=obj_name)
        self._attention   = attention
        self._feedforward = feedforward
    
    @property
    def attention(self) -> SelfAttention:
        return self._attention

    @property
    def feedforward(self) -> FeedForward:
        return self._feedforward

    @classmethod
    def create_via_class(cls, 
                         obj_name: str, 
                         embedding_size: int, 
                         head_number: int, 
                         scale_rate: int, 
                         attention_cls: SelfAttention,
                         feedforward_cls: FeedForward) -> TransformerEncoderBlock:
        return cls(attention_cls(obj_name="attention", embedding_size=embedding_size, head_number=head_number), 
                   feedforward_cls(obj_name="feedforward", embedding_size=embedding_size, scale_rate=scale_rate))

    def setting(self) -> NoReturn: 
        self.assign(name="attention", component=self.attention)
        self.assign(name="feedforward", component=self.feedforward)
    
    def model(self, input_data: tf.Tensor, mask: tf.Tensor=None) -> tf.Tensor:
        layers = [self.assigned.attention(mask=mask), self.assigned.feedforward]
        output = input_data
        for layer in layers:
            output = layer(input_data=output)
        return output

    def __call__(self, input_data: tf.Tensor, mask: tf.Tensor=None) -> tf.Tensor:
        return self.model(input_data, mask)

def transformer(obj_name: str, embedding_size: int, head_number: int, scale_rate: int) -> Block:
    attention = SelfAttention("attention", embedding_size=embedding_size, head_number=head_number)
    feedforward = FeedForward("feedforward", embedding_size=embedding_size, scale_rate=scale_rate)
    block = TransformerEncoderBlock(obj_name=obj_name, 
                                    attention=attention, 
                                    feedforward=feedforward)
    return block