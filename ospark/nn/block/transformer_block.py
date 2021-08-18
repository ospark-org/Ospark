from __future__ import annotations
from typing import Optional, NoReturn
from ospark.nn.layer.self_attention import SelfAttention, EncoderDecoderAttention
from ospark.nn.layer.feed_forward import FeedForward
from . import Block
import tensorflow as tf


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
        return cls(obj_name=obj_name, 
                   attention=attention_cls(obj_name="attention", embedding_size=embedding_size, head_number=head_number), 
                   feedforward=feedforward_cls(obj_name="feedforward", embedding_size=embedding_size, scale_rate=scale_rate))

    def initialize(self) -> NoReturn:
        self.assign(name="attention", component=self.attention)
        self.assign(name="feedforward", component=self.feedforward)
    
    def model(self, input_data: tf.Tensor, mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        layers = [self.assigned.attention(mask=mask), self.assigned.feedforward]
        output = input_data
        for layer in layers:
            output = layer(input_data=output)
        return output

    def __call__(self, input_data: tf.Tensor, mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        return self.model(input_data, mask)


class TransformerDecoderBlock(Block):

    def __init__(self,
                 obj_name: str, 
                 attention: SelfAttention,
                 encode_decode_attention: EncoderDecoderAttention,
                 feedforward: FeedForward) -> NoReturn:
        super().__init__(obj_name)
        self._attention               = attention
        self._encode_decode_attention = encode_decode_attention
        self._feedforward             = feedforward
    
    @property
    def attention(self) -> SelfAttention:
        return self._attention

    @property
    def encode_decode_attention(self) -> EncoderDecoderAttention:
        return self._encode_decode_attention

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
                         encode_decode_attention_cls: EncoderDecoderAttention,
                         feedforward_cls: FeedForward) -> TransformerEncoderBlock:
        return cls(obj_name=obj_name, 
                   attention=attention_cls(obj_name="attention", embedding_size=embedding_size, head_number=head_number, look_ahead=True),
                   encode_decode_attention=encode_decode_attention_cls(obj_name="encode_decode_attention", embedding_size=embedding_size, head_number=head_number),
                   feedforward=feedforward_cls(obj_name="feedforward", embedding_size=embedding_size, scale_rate=scale_rate))

    def initialize(self) -> NoReturn:
        self.assign(name="attention", component=self.attention)
        self.assign(name="encode_decode_attention", component=self.encode_decode_attention)
        self.assign(name="feedforward", component=self.feedforward)

    def model(self, input_data: tf.Tensor, encoder_output: tf.Tensor, mask: Optional[tf.Tensor]) -> tf.Tensor:
        layers = [self.assigned.attention(mask), 
                  self.assigned.encode_decode_attention(mask, encoder_output), 
                  self.assigned.feedforward]
        output = input_data
        for layer in layers:
            output = layer(output)
        return output

    def __call__(self, input_data: tf.Tensor, encoder_output: tf.Tensor, mask: Optional[tf.Tensor]) -> tf.Tensor:
        return self.model(input_data, encoder_output, mask)


def encoder_block(obj_name: str, embedding_size: int, head_number: int, scale_rate: int) -> Block:
    attention = SelfAttention("attention", embedding_size=embedding_size, head_number=head_number)
    feedforward = FeedForward("feedforward", embedding_size=embedding_size, scale_rate=scale_rate)
    block = TransformerEncoderBlock(obj_name=obj_name, 
                                    attention=attention, 
                                    feedforward=feedforward)
    return block

def decoder_block(obj_name: str, embedding_size: int, head_number: int, scale_rate: int) -> Block:
    attention = SelfAttention("attention", embedding_size=embedding_size, head_number=head_number, look_ahead=True)
    encode_decode_attention = EncoderDecoderAttention("encode_decode_attention", embedding_size=embedding_size, head_number=head_number)
    feedforward = FeedForward("feedforward", embedding_size=embedding_size, scale_rate=scale_rate)
    block = TransformerDecoderBlock(obj_name=obj_name, 
                                    attention=attention, 
                                    encode_decode_attention=encode_decode_attention,
                                    feedforward=feedforward)
    return block