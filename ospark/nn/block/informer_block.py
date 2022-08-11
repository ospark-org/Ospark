from __future__ import annotations
from typing import NoReturn, Optional, Type
from ospark.nn.layers.self_attention import SelfAttentionLayer, EncoderDecoderAttentionLayer
from ospark.nn.layers.prob_sparse import ProbSparseAttentionLayer
from ospark.nn.layers.distilling_layer import DistillingLayer
from ospark.nn.layers.feed_forward import FeedForwardLayer
from ospark.nn.component.activation import GELU
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
                         attention_cls: Type[SelfAttentionLayer],
                         feedforward_cls: Type[FeedForwardLayer],
                         distilling_cls: Type[DistillingLayer],
                         is_training: Optional[bool]=None) -> InformerEncoderBlock:
        return cls(obj_name=obj_name,
                   attention=attention_cls(obj_name="attention",
                                           embedding_size=embedding_size,
                                           head_number=head_number,
                                           is_training=is_training),
                   feedforward=feedforward_cls(obj_name="feedforward",
                                               embedding_size=embedding_size,
                                               scale_rate=scale_rate,
                                               is_training=is_training),
                   distilling=distilling_cls(obj_name="distilling",
                                             embedding_size=embedding_size,
                                             is_training=is_training))

    def in_creating(self) -> NoReturn:
        self.assign(component=self.attention, name="attention")
        self.assign(component=self.feedforward, name="feedforward")
        self.assign(component=self.distilling, name="distilling")

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        layers = [self.assigned.attention(None),
                  self.assigned.feedforward,
                  self.assigned.distilling]
        output = input_data
        for layer in layers:
            output = layer.pipeline(output)
        return output


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

    def in_creating(self) -> NoReturn:
        self.assign(component=self.attention, name="attention")
        self.assign(component=self.encode_decode_attention, name="encode_decode_attention")
        self.assign(component=self.feedforward, name="feedforward")

    def pipeline(self, input_data: tf.Tensor, encoder_output: tf.Tensor) -> tf.Tensor:
        layers = [self.assigned.attention(None),
                  self.assigned.encode_decode_attention(mask=None, encoder_output=encoder_output),
                  self.feedforward]
        output = input_data
        for layer in layers:
            output = layer(output)
        return output


def informer_encoder_block(obj_name: str,
                           embedding_size: int,
                           head_number: int,
                           scale_rate: int,
                           sample_factor: float,
                           dropout_rate: float,
                           is_training: Optional[bool]=None,
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
                                   activation=GELU(),
                                   dropout_rate=dropout_rate,
                                   is_training=is_training)
    distilling  = DistillingLayer("distilling",
                                  embedding_size=embedding_size,
                                  filter_width=filter_width,
                                  pooling_size=pooling_size,
                                  strides=strides,
                                  is_training=is_training)
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
                                   activation=GELU(),
                                   dropout_rate=dropout_rate,
                                   is_training=is_training)
    block = InformerDecoderBlock(obj_name=obj_name,
                                 attention=attention,
                                 encode_decode_attention=encode_decode_attention,
                                 feedforward=feedforward)
    return block

