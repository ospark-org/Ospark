from __future__ import annotations
from typing import Optional, NoReturn, Callable, List
from ospark.nn.layer.self_attention import SelfAttentionLayer, EncoderDecoderAttentionLayer
from ospark.nn.layer.deep_attention import DeepAttentionLayer, DeepEncoderDecoder
from ospark.nn.layer.feed_forward import FeedForwardLayer
from ospark.nn.layer.deep_feed_forward import DeepFeedForwardLayer
from . import Block
import tensorflow as tf


class TransformerEncoderBlock(Block):

    def __init__(self,
                 obj_name: str,
                 attention: SelfAttentionLayer,
                 feedforward: FeedForwardLayer) -> NoReturn:
        super().__init__(obj_name=obj_name)
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
                         attention_cls: SelfAttentionLayer,
                         feedforward_cls: FeedForwardLayer,
                         dropout_rate: float,
                         is_training: Optional[bool]=False) -> TransformerEncoderBlock:
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
                                               is_training=is_training))

    def on_creating(self) -> NoReturn:
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

    @classmethod
    def create_via_class(cls,
                         obj_name: str,
                         embedding_size: int,
                         head_number: int,
                         scale_rate: int,
                         attention_cls: SelfAttentionLayer,
                         encode_decode_attention_cls: EncoderDecoderAttentionLayer,
                         feedforward_cls: FeedForwardLayer,
                         dropout_rate: float,
                         is_training: Optional[bool]=False) -> TransformerEncoderBlock:
        return cls(obj_name=obj_name, 
                   attention=attention_cls(obj_name="attention",
                                           embedding_size=embedding_size,
                                           head_number=head_number,
                                           look_ahead=True,
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
                                               is_training=is_training))

    def on_creating(self) -> NoReturn:
        self.assign(name="attention", component=self.attention)
        self.assign(name="encode_decode_attention", component=self.encode_decode_attention)
        self.assign(name="feedforward", component=self.feedforward)

    def model(self,
              input_data: tf.Tensor,
              encoder_output: tf.Tensor,
              encoder_padding_mask: Optional[tf.Tensor],
              decoder_padding_mask: Optional[tf.Tensor]) -> tf.Tensor:
        layers = [self.assigned.attention(decoder_padding_mask),
                  self.assigned.encode_decode_attention(encoder_padding_mask, encoder_output),
                  self.assigned.feedforward]
        output = input_data
        for layer in layers:
            output = layer(output)
        return output

    def __call__(self,
                 input_data: tf.Tensor,
                 encoder_output: tf.Tensor,
                 encoder_padding_mask: Optional[tf.Tensor],
                 decoder_padding_mask: Optional[tf.Tensor]) -> tf.Tensor:
        return self.model(input_data, encoder_output, encoder_padding_mask, decoder_padding_mask)


def transformer_encoder_block(obj_name: str,
                              embedding_size: int,
                              head_number: int,
                              scale_rate: int,
                              dropout_rate: float,
                              is_training: Optional[bool]=False) -> Block:
    attention = SelfAttentionLayer(obj_name="attention",
                                   embedding_size=embedding_size,
                                   head_number=head_number,
                                   dropout_rate=dropout_rate,
                                   is_training=is_training)
    feedforward = FeedForwardLayer(obj_name="feedforward",
                                   embedding_size=embedding_size,
                                   scale_rate=scale_rate,
                                   dropout_rate=dropout_rate,
                                   is_training=is_training)
    block = TransformerEncoderBlock(obj_name=obj_name, 
                                    attention=attention, 
                                    feedforward=feedforward)
    return block

def transformer_decoder_block(obj_name: str,
                              embedding_size: int,
                              head_number: int,
                              scale_rate: int,
                              dropout_rate: float,
                              is_training: Optional[bool]=False) -> Block:
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
                                    feedforward=feedforward)
    return block

def create_coder_blocks(block_number: int,
                        create_func: Callable[[str, int, int, int, float], Block],
                        embedding_size: int,
                        head_number: int,
                        scale_rate: int,
                        dropout_rate: Optional[float]=0.0) -> List[Block]:
    blocks = []
    for i in range(block_number):
        name = f"block_{i}"
        blocks.append(create_func(obj_name=name,
                                  embedding_size=embedding_size,
                                  head_number=head_number,
                                  scale_rate=scale_rate,
                                  dropout_rate=dropout_rate))
    return blocks

