from __future__ import annotations
from typing import NoReturn
from ospark.nn.layer.self_attention import SelfAttention, EncoderDecoderAttention
from ospark.nn.layer.prob_sparse import ProbSparse
from ospark.nn.layer.distilling_layer import DistillingLayer
from . import Block
import tensorflow as tf

class InformerEncoderBlock(Block):

    def __init__(self, 
                 obj_name: str,
                 attention: SelfAttention,
                 distilling: DistillingLayer) -> NoReturn:
        super().__init__(obj_name)
        self._attention      = attention
        self._distilling     = distilling
 
    @property
    def attention(self) -> SelfAttention:
        return self._attention

    @property
    def distilling(self) -> DistillingLayer:
        return self._distilling

    @classmethod
    def create_via_class(cls, 
                         obj_name: str, 
                         embedding_size: int, 
                         head_number: int, 
                         attention_cls: SelfAttention,
                         distilling_cls: DistillingLayer) -> InformerEncoderBlock:
        return cls(obj_name=obj_name,
                   attention=attention_cls(obj_name="attention", embedding_size=embedding_size, head_number=head_number),
                   distilling=distilling_cls(obj_name="distilling", embedding_size=embedding_size))

    def setting(self) -> NoReturn:
        self.assign(component=self.attention, name="attention")
        self.assign(component=self.distilling, name="distilling")

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        layers = [self.assigned.attention, self.assigned.distilling]
        output = input_data
        for layer in layers:
            output = layer(output)
        return output

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        return self.model(input_data)

class InformerDecoderBlock(Block):

    def __init__(self, 
                 obj_name: str,
                 attention: SelfAttention,
                 encode_decode_attention: EncoderDecoderAttention) -> NoReturn:
        super().__init__(obj_name)
        self._attention               = attention
        self._encode_decode_attention = encode_decode_attention

    @property
    def attention(self) -> SelfAttention:
        return self._attention

    @property
    def encode_decode_attention(self) -> EncoderDecoderAttention:
        return self._encode_decode_attention

    def setting(self) -> NoReturn:
        self.assign(component=self.attention, name="attention")
        self.assign(component=self.encode_decode_attention, name="encode_decode_attention")

    def model(self, input_data: tf.Tensor, encoder_output: tf.Tensor) -> tf.Tensor:
        layers = [self.assigned.attention, self.assigned.encode_decode_attention(mask=None, encoder_output=encoder_output)]
        output = input_data
        for layer in layers:
            output = layer(output)
        return output

    def __call__(self, input_data: tf.Tensor, encoder_output: tf.Tensor) -> tf.Tensor:
        return self.model(input_data, encoder_output)

def encoder_block(obj_name: str, embedding_size: int, head_number: int, sample_factor: float,
                  filter_width: int=None, pooling_size: list=None, strides: list=None) -> Block:
    attention = ProbSparse("attention", embedding_size=embedding_size, head_number=head_number, sample_factor=sample_factor)
    distilling = DistillingLayer("distilling", embedding_size=embedding_size, filter_width=filter_width, pooling_size=pooling_size, strides=strides)
    block = InformerEncoderBlock(obj_name=obj_name,
                                 attention=attention,
                                 distilling=distilling)
    return block

def decoder_block(obj_name: str, embedding_size: int, head_number: int, sample_factor: float) -> Block:
    attention = ProbSparse("sparse_attention", embedding_size=embedding_size, head_number=head_number, sample_factor=sample_factor, look_ahead=True)
    encode_decode_attention = EncoderDecoderAttention("encode_decode_attention", embedding_size=embedding_size, head_number=head_number)
    block = InformerDecoderBlock(obj_name=obj_name,
                                 attention=attention,
                                 encode_decode_attention=encode_decode_attention)
    return block

