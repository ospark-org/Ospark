from __future__ import annotations
from typing import Optional, NoReturn, Callable, List, Type
from ospark.nn.layers.self_attention import SelfAttentionLayer, EncoderDecoderAttentionLayer
from ospark.nn.layers.deep_attention import DeepAttentionLayer, DeepEncoderDecoder
from ospark.nn.layers.feed_forward import FeedForwardLayer
from ospark.nn.layers.deep_feed_forward import DeepFeedForwardLayer
from . import Block
import tensorflow as tf
import ospark


class TransformerEncoderBlock(Block):

    def __init__(self,
                 obj_name: str,
                 feedforward: FeedForwardLayer,
                 attention: SelfAttentionLayer,
                 embedding_size: int,
                 is_training: Optional[bool]=None,
                 training_phase: Optional[bool]=None,
                 use_norm_output: Optional[bool]=None) -> NoReturn:
        """
        Args:
            obj_name:
            feedforward:
            attention:
            is_training:
            training_phase:
            use_norm_output:
                Has two modes "pre-ln" and "post-ln", default is "pre-ln"
        """

        super().__init__(obj_name=obj_name, is_training=is_training, training_phase=training_phase)
        self._attention       = attention
        self._feedforward     = feedforward
        self._use_norm_output = use_norm_output if use_norm_output is not None else True
        self._embedding_size  = embedding_size
    
    @property
    def attention(self) -> SelfAttentionLayer:
        return self._attention

    @property
    def feedforward(self) -> FeedForwardLayer:
        return self._feedforward

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @classmethod
    def create_via_class(cls,
                         obj_name: str,
                         embedding_size: int,
                         head_number: int,
                         scale_rate: int,
                         attention_cls: Type[SelfAttentionLayer],
                         feedforward_cls: Type[FeedForwardLayer],
                         dropout_rate: float,
                         is_training: Optional[bool]=None) -> TransformerEncoderBlock:
        return cls(obj_name=obj_name, 
                   attention=attention_cls(obj_name="attention",
                                           embedding_size=embedding_size,
                                           head_number=head_number,
                                           dropout_rate=dropout_rate,
                                           is_training=is_training),
                   feedforward=feedforward_cls(obj_name="feedforward",
                                               embedding_size=embedding_size,
                                               scale_rate=scale_rate,
                                               dropout_rate=dropout_rate,
                                               is_training=is_training),
                   embedding_size=embedding_size)

    def in_creating(self) -> NoReturn:
        self.assign(name="attention", component=self.attention)
        self.assign(name="feedforward", component=self.feedforward)
        if self._use_norm_output:
            self._norm = ospark.nn.layers.normalization.LayerNormalization(obj_name="layer_norm",
                                                                           layer_dimension=self.embedding_size)
    
    def pipeline(self, input_data: tf.Tensor, mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        output = self.assigned.attention.pipeline(input_data=input_data, mask=mask)
        output = self.assigned.feedforward.pipeline(input_data=output)
        if self._use_norm_output:
            output = self._norm(output)
        return output


class TransformerDecoderBlock(Block):

    def __init__(self,
                 obj_name: str,
                 attention: SelfAttentionLayer,
                 encode_decode_attention: EncoderDecoderAttentionLayer,
                 feedforward: FeedForwardLayer,
                 embedding_size: int,
                 is_training: Optional[bool]=None,
                 training_phase: Optional[bool]=None,
                 use_norm_output: Optional[bool]=None) -> NoReturn:
        super().__init__(obj_name=obj_name, is_training=is_training, training_phase=training_phase)
        self._attention               = attention
        self._encode_decode_attention = encode_decode_attention
        self._feedforward             = feedforward
        self._use_norm_output         = use_norm_output if use_norm_output is not None else True
        self._embedding_size          = embedding_size
    
    @property
    def attention(self) -> SelfAttentionLayer:
        return self._attention

    @property
    def encode_decode_attention(self) -> EncoderDecoderAttentionLayer:
        return self._encode_decode_attention

    @property
    def feedforward(self) -> FeedForwardLayer:
        return self._feedforward

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @classmethod
    def create_via_class(cls,
                         obj_name: str,
                         embedding_size: int,
                         head_number: int,
                         scale_rate: int,
                         attention_cls: Type[SelfAttentionLayer],
                         encode_decode_attention_cls: Type[EncoderDecoderAttentionLayer],
                         feedforward_cls: Type[FeedForwardLayer],
                         dropout_rate: float,
                         is_training: Optional[bool]=None) -> TransformerDecoderBlock:
        return cls(obj_name=obj_name, 
                   attention=attention_cls(obj_name="attention",
                                           embedding_size=embedding_size,
                                           head_number=head_number,
                                           use_look_ahead=True,
                                           dropout_rate=dropout_rate,
                                           is_training=is_training),
                   encode_decode_attention=encode_decode_attention_cls(obj_name="encode_decode_attention",
                                                                       embedding_size=embedding_size,
                                                                       head_number=head_number,
                                                                       dropout_rate=dropout_rate,
                                                                       is_training=is_training),
                   feedforward=feedforward_cls(obj_name="feedforward",
                                               embedding_size=embedding_size,
                                               scale_rate=scale_rate,
                                               dropout_rate=dropout_rate,
                                               is_training=is_training),
                   embedding_size=embedding_size)

    def in_creating(self) -> NoReturn:
        self.assign(name="attention", component=self.attention)
        self.assign(name="encode_decode_attention", component=self.encode_decode_attention)
        self.assign(name="feedforward", component=self.feedforward)
        if self._use_norm_output:
            self._norm = ospark.nn.layers.normalization.LayerNormalization(obj_name="layer_norm",
                                                                           layer_dimension=self.embedding_size)

    def pipeline(self,
                 input_data: tf.Tensor,
                 encoder_output: tf.Tensor,
                 encoder_padding_mask: tf.Tensor,
                 decoder_padding_mask: tf.Tensor) -> tf.Tensor:
        output = self.assigned.attention.pipeline(input_data=input_data, mask=decoder_padding_mask)
        output = self.assigned.encode_decode_attention.pipeline(input_data=output,
                                                                mask=encoder_padding_mask,
                                                                encoder_output=encoder_output)
        output = self.assigned.feedforward.pipeline(input_data=output)
        if self._use_norm_output:
            output = self._norm(output)
        return output


def transformer_encoder_block(obj_name: str,
                              embedding_size: int,
                              head_number: int,
                              scale_rate: int,
                              dropout_rate: float,
                              mode: Optional[str]=None,
                              use_norm_output: Optional[bool]=None,
                              is_training: Optional[bool]=None) -> Block:
    attention = SelfAttentionLayer(obj_name="attention",
                                   embedding_size=embedding_size,
                                   head_number=head_number,
                                   dropout_rate=dropout_rate,
                                   is_training=is_training,
                                   mode=mode)
    feedforward = FeedForwardLayer(obj_name="feedforward",
                                   embedding_size=embedding_size,
                                   scale_rate=scale_rate,
                                   dropout_rate=dropout_rate,
                                   is_training=is_training,
                                   mode=mode)
    block = TransformerEncoderBlock(obj_name=obj_name, feedforward=feedforward, attention=attention,
                                    is_training=is_training, embedding_size=embedding_size, use_norm_output=use_norm_output)
    return block

def transformer_decoder_block(obj_name: str,
                              embedding_size: int,
                              head_number: int,
                              scale_rate: int,
                              dropout_rate: float,
                              use_norm_output: Optional[bool]=None,
                              is_training: Optional[bool]=None) -> Block:
    attention = SelfAttentionLayer(obj_name="attention",
                                   embedding_size=embedding_size,
                                   head_number=head_number,
                                   dropout_rate=dropout_rate,
                                   is_training=is_training,
                                   use_look_ahead=True)
    encode_decode_attention = EncoderDecoderAttentionLayer(obj_name="encode_decode_attention",
                                                           embedding_size=embedding_size,
                                                           head_number=head_number,
                                                           dropout_rate=dropout_rate,
                                                           is_training=is_training)
    feedforward = FeedForwardLayer(obj_name="feedforward",
                                   embedding_size=embedding_size,
                                   scale_rate=scale_rate,
                                   dropout_rate=dropout_rate,
                                   is_training=is_training)
    block = TransformerDecoderBlock(obj_name=obj_name, 
                                    attention=attention, 
                                    encode_decode_attention=encode_decode_attention,
                                    feedforward=feedforward,
                                    embedding_size=embedding_size,
                                    use_norm_output=use_norm_output)
    return block

from ospark.nn.layers.self_attention import FavorAttentionLayer, EncodeDecodeFavorAttention


def performer_encoder_block(obj_name: str,
                            embedding_size: int,
                            head_number: int,
                            scale_rate: int,
                            dropout_rate: float,
                            random_projections_number: int,
                            is_training: Optional[bool]=None) -> Block:
    attention = FavorAttentionLayer(obj_name="favor_attention",
                                    embedding_size=embedding_size,
                                    head_number=head_number,
                                    dropout_rate=dropout_rate,
                                    is_training=is_training,
                                    random_projections_number=random_projections_number)
    feedforward = FeedForwardLayer(obj_name="feedforward",
                                   embedding_size=embedding_size,
                                   scale_rate=scale_rate,
                                   dropout_rate=dropout_rate,
                                   is_training=is_training)
    block = TransformerEncoderBlock(obj_name=obj_name, feedforward=feedforward, attention=attention)
    return block


def performer_decoder_block(obj_name: str,
                            embedding_size: int,
                            head_number: int,
                            scale_rate: int,
                            dropout_rate: float,
                            random_projections_number: int,
                            is_training: Optional[bool]=None) -> Block:
    attention = FavorAttentionLayer(obj_name="favor_attention",
                                    embedding_size=embedding_size,
                                    head_number=head_number,
                                    dropout_rate=dropout_rate,
                                    is_training=is_training,
                                    use_look_ahead=True,
                                    random_projections_number=random_projections_number)
    encode_decode_attention = EncodeDecodeFavorAttention(obj_name="encode_decode_favor_attention",
                                                         embedding_size=embedding_size,
                                                         head_number=head_number,
                                                         dropout_rate=dropout_rate,
                                                         is_training=is_training,
                                                         random_projections_number=random_projections_number)
    feedforward = FeedForwardLayer(obj_name="feedforward",
                                   embedding_size=embedding_size,
                                   scale_rate=scale_rate,
                                   dropout_rate=dropout_rate,
                                   is_training=is_training)
    block = TransformerDecoderBlock(obj_name=obj_name,
                                    attention=attention,
                                    encode_decode_attention=encode_decode_attention,
                                    feedforward=feedforward)
    return block