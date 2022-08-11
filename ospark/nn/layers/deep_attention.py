from ospark.nn.layers.normalization import Normalization
from ospark.nn.layers.self_attention import SelfAttentionLayer, EncoderDecoderAttentionLayer
from typing import NoReturn, Optional, Union
import tensorflow as tf


class DeepAttentionLayer(SelfAttentionLayer):

    def __init__(self,
                 obj_name: str,
                 embedding_size: int,
                 head_number: int,
                 dropout_rate: float,
                 is_training: Optional[bool]=None,
                 use_look_ahead: Optional[bool]=False) -> NoReturn:
        super().__init__(obj_name=obj_name,
                         embedding_size=embedding_size,
                         head_number=head_number,
                         dropout_rate=dropout_rate,
                         is_training=is_training,
                         use_look_ahead=use_look_ahead)
        self._res_adaptive_value = 1

    @property
    def res_adaptive_value(self) -> Union[int, tf.Tensor]:
        return self._res_adaptive_value

    def profiling_phase(self, input_data: tf.Tensor, catch_variance: list, mask: Optional[tf.Tensor]=None):
        self._res_adaptive_value = 1.

        output      = self.__call__(mask=mask)(input_data)
        _, variance = tf.nn.moments(output, axes=[0, 1])
        catch_variance.append(variance)

        self._res_adaptive_value = tf.sqrt(tf.reduce_sum(catch_variance, axis=0))[tf.newaxis, tf.newaxis, :]
        return output, catch_variance

    def residual_net(self, input_data: tf.Tensor) -> tf.Tensor:
        output = tf.multiply(input_data, self.res_adaptive_value)
        return output

    def back_to_standard(self) -> NoReturn:
        self._res_adaptive_value = 1


class DeepEncoderDecoder(EncoderDecoderAttentionLayer):

    def __init__(self,
                 obj_name: str,
                 embedding_size: int,
                 head_number: int,
                 dropout_rate: float,
                 is_training: Optional[bool]=None,
                 use_look_ahead: Optional[bool]=False) -> NoReturn:
        super().__init__(obj_name=obj_name,
                         embedding_size=embedding_size,
                         head_number=head_number,
                         dropout_rate=dropout_rate,
                         is_training=is_training,
                         use_look_ahead=use_look_ahead)
        self._res_adaptive_value = 1

    @property
    def res_adaptive_value(self) -> Union[int, tf.Tensor]:
        return self._res_adaptive_value

    def profiling_phase(self, input_data: tf.Tensor, encoder_output: tf.Tensor, catch_variance: list, mask: Optional[tf.Tensor]=None):
        self._res_adaptive_value = 1.

        output   = self.__call__(mask=mask, encoder_output=encoder_output)(input_data)
        _, variance = tf.nn.moments(output, axes=[0, 1])
        catch_variance.append(variance)

        self._res_adaptive_value = tf.sqrt(tf.reduce_sum(catch_variance, axis=0))[tf.newaxis, tf.newaxis, :]
        return output, catch_variance

    def residual_net(self, input_data: tf.Tensor) -> tf.Tensor:
        output = tf.multiply(input_data, self.res_adaptive_value)
        return output

    def back_to_standard(self) -> NoReturn:
        self._res_adaptive_value = 1.

