from __future__ import annotations
from typing import NoReturn, Optional
from ospark.nn.layer.self_attention import SelfAttentionLayer, EncoderDecoderAttentionLayer
from ospark.nn.layer.prob_sparse import ProbSparseAttentionLayer
from ospark.nn.layer.distilling_layer import DistillingLayer
from ospark.nn.layer.feed_forward import FeedForwardLayer
from ospark.nn.component.activation import gelu
from . import Block
import tensorflow as tf

class InformerEncoderBlock(Block):

    def __init__(self,
                 obj_name: str,
                 attention: SelfAttentionLayer,
                 feedforward: FeedForwardLayer,
                 distilling: DistillingLayer) -> NoReturn:
        super().__init__(obj_name)
        self._attention      = attention
        self._feedforward    = feedforward
        self._distilling     = distilling
 
    @property
    def attention(self) -> SelfAttentionLayer:
        return self._attention

    @property
    def feedforward(self) -> FeedForwardLayer:
        return self._feedforward

    @property
    def distilling(self) -> DistillingLayer:
        return self._distilling

    @classmethod
    def create_via_class(cls,
                         obj_name: str,
                         embedding_size: int,
                         head_number: int,
                         scale_rate: int,
                         attention_cls: SelfAttentionLayer,
                         feedforward_cls: FeedForwardLayer,
                         distilling_cls: DistillingLayer) -> InformerEncoderBlock:
        return cls(obj_name=obj_name,
                   attention=attention_cls(obj_name="attention", embedding_size=embedding_size, head_number=head_number),
                   feedforward=feedforward_cls(obj_name="feedforward", embedding_size=embedding_size, scale_rate=scale_rate),
                   distilling=distilling_cls(obj_name="distilling", embedding_size=embedding_size))

    def on_creating(self) -> NoReturn:
        self.assign(component=self.attention, name="attention")
        self.assign(component=self.feedforward, name="feedforward")
        self.assign(component=self.distilling, name="distilling")

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        layers = [self.assigned.attention(None),
                  self.assigned.feedforward,
                  self.assigned.distilling]
        output = input_data
        for layer in layers:
            output = layer(output)
        return output

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        return self.model(input_data)

class InformerDecoderBlock(Block):

    def __init__(self,
                 obj_name: str,
                 attention: SelfAttentionLayer,
                 encode_decode_attention: EncoderDecoderAttentionLayer,
                 feedforward: FeedForwardLayer) -> NoReturn:
        super().__init__(obj_name)
        self._attention               = attention
        self._encode_decode_attention = encode_decode_attention
        self._feedforward             = feedforward

    @property
    def attention(self) -> SelfAttentionLayer:
        return self._attention

    @property
    def encode_decode_attention(self) -> EncoderDecoderAttentionLayer:
        return self._encode_decode_attention

    @property
    def feedforward(self) -> FeedForwardLayer:
        return self._feedforward

    def on_creating(self) -> NoReturn:
        self.assign(component=self.attention, name="attention")
        self.assign(component=self.encode_decode_attention, name="encode_decode_attention")
        self.assign(component=self.feedforward, name="feedforward")

    def model(self, input_data: tf.Tensor, encoder_output: tf.Tensor) -> tf.Tensor:
        layers = [self.assigned.attention(None),
                  self.assigned.encode_decode_attention(mask=None, encoder_output=encoder_output),
                  self.feedforward]
        output = input_data
        for layer in layers:
            output = layer(output)
        return output

    def __call__(self, input_data: tf.Tensor, encoder_output: tf.Tensor) -> tf.Tensor:
        return self.model(input_data, encoder_output)

def informer_encoder_block(obj_name: str,
                           embedding_size: int,
                           head_number: int,
                           scale_rate: int,
                           sample_factor: float,
                           dropout_rate: float,
                           is_training: Optional[bool]=False,
                           filter_width: int=None,
                           pooling_size: list=None,
                           strides: list=None) -> Block:
    attention   = ProbSparseAttentionLayer("attention",
                                           embedding_size=embedding_size,
                                           head_number=head_number,
                                           sample_factor=sample_factor,
                                           dropout_rate=dropout_rate,
                                           is_training=is_training)
    feedforward = FeedForwardLayer("feedforward",
                                   embedding_size=embedding_size,
                                   scale_rate=scale_rate,
                                   activation=gelu(),
                                   dropout_rate=dropout_rate,
                                   is_training=is_training)
    distilling  = DistillingLayer("distilling",
                                  embedding_size=embedding_size,
                                  filter_width=filter_width,
                                  pooling_size=pooling_size,
                                  strides=strides)
    block = InformerEncoderBlock(obj_name=obj_name,
                                 attention=attention,
                                 feedforward=feedforward,
                                 distilling=distilling)
    return block

def informer_decoder_block(obj_name: str,
                           embedding_size: int,
                           head_number: int,
                           scale_rate: int,
                           sample_factor: float,
                           dropout_rate: float,
                           is_training: Optional[bool]=False) -> Block:
    attention = ProbSparseAttentionLayer("sparse_attention",
                                         embedding_size=embedding_size,
                                         head_number=head_number,
                                         sample_factor=sample_factor,
                                         use_look_ahead=True,
                                         dropout_rate=dropout_rate,
                                         is_training=is_training)
    encode_decode_attention = EncoderDecoderAttentionLayer("encode_decode_attention",
                                                           embedding_size=embedding_size,
                                                           head_number=head_number,
                                                           dropout_rate=dropout_rate,
                                                           is_training=is_training)
    feedforward = FeedForwardLayer("feedforward",
                                   embedding_size=embedding_size,
                                   scale_rate=scale_rate,
                                   activation=gelu(),
                                   dropout_rate=dropout_rate,
                                   is_training=is_training)
    block = InformerDecoderBlock(obj_name=obj_name,
                                 attention=attention,
                                 encode_decode_attention=encode_decode_attention,
                                 feedforward=feedforward)
    return block

