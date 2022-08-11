from ospark.nn.block.transformer_block import TransformerEncoderBlock, TransformerDecoderBlock
from ospark.nn.layers.deep_attention import DeepEncoderDecoder, DeepAttentionLayer
from ospark.nn.layers.deep_feed_forward import DeepFeedForwardLayer
from typing import Optional, NoReturn
import tensorflow as tf


class DeepTransformerEncoder(TransformerEncoderBlock):

    def profiling_phase(self, input_data: tf.Tensor, catch_variance: list, mask: Optional[tf.Tensor]=None):
        output, catch_variance = self.assigned.attention.profiling_phase(input_data=input_data,
                                                                         catch_variance=catch_variance,
                                                                         mask=mask)
        output, catch_variance = self.assigned.feedforward.profiling_phase(input_data=input_data,
                                                                           catch_variance=catch_variance)
        return output, catch_variance

    def recover_transformer(self) -> NoReturn:
        self.assigned.attention.back_to_standard()
        self.assigned.feedforward.back_to_standard()


class DeepTransformerDecoder(TransformerDecoderBlock):

    def profiling_phase(self,
                        input_data: tf.Tensor,
                        encoder_output: tf.Tensor,
                        catch_variance: list,
                        encoder_padding_mask: Optional[tf.Tensor]=None,
                        decoder_padding_mask: Optional[tf.Tensor]=None
                        ):
        output, catch_variance = self.assigned.attention.profiling_phase(input_data=input_data,
                                                                         catch_variance=catch_variance,
                                                                         mask=decoder_padding_mask)
        output, catch_variance = self.assigned.encode_decode_attention.profiling_phase(input_data=output,
                                                                                       encoder_output=encoder_output,
                                                                                       catch_variance=catch_variance,
                                                                                       mask=encoder_padding_mask)
        output, catch_variance = self.assigned.feedforward.profiling_phase(input_data=output,
                                                                           catch_variance=catch_variance)
        return output, catch_variance

    def recover_transformer(self) -> NoReturn:
        self.assigned.attention.back_to_standard()
        self.assigned.encode_decode_attention.back_to_standard()
        self.assigned.feedforward.back_to_standard()


def exdeep_encoder_block(obj_name:str,
                         embedding_size: int,
                         head_number: int,
                         scale_rate: int,
                         dropout_rate: float,
                         is_training: Optional[bool] = False
                         ) -> DeepTransformerEncoder:
    attention_layer = DeepAttentionLayer(obj_name="attention",
                                         embedding_size=embedding_size,
                                         head_number=head_number,
                                         dropout_rate=dropout_rate,
                                         is_training=is_training)
    feedforward_layer = DeepFeedForwardLayer(obj_name="feedforward",
                                             embedding_size=embedding_size,
                                             scale_rate=scale_rate,
                                             dropout_rate=dropout_rate,
                                             is_training=is_training)
    block = DeepTransformerEncoder(obj_name=obj_name, attention=attention_layer, feedforward=feedforward_layer)
    return block

def exdeep_decoder_block(obj_name:str,
                         embedding_size: int,
                         head_number: int,
                         scale_rate:int,
                         dropout_rate: float,
                         is_training: Optional[bool] = False
                         ) -> DeepTransformerDecoder:
    attention_layer = DeepAttentionLayer(obj_name="attention",
                                         embedding_size=embedding_size,
                                         head_number=head_number,
                                         use_look_ahead=True,
                                         dropout_rate=dropout_rate,
                                         is_training=is_training)
    encoder_deocder_layer = DeepEncoderDecoder(obj_name="encode_decode_attention",
                                               embedding_size=embedding_size,
                                               head_number=head_number,
                                               dropout_rate=dropout_rate,
                                               is_training=is_training)
    feedforward_layer = DeepFeedForwardLayer(obj_name="feedforward",
                                             embedding_size=embedding_size,
                                             scale_rate=scale_rate,
                                             dropout_rate=dropout_rate,
                                             is_training=is_training)
    block = DeepTransformerDecoder(obj_name=obj_name,
                                   attention=attention_layer,
                                   encode_decode_attention=encoder_deocder_layer,
                                   feedforward=feedforward_layer)
    return block

