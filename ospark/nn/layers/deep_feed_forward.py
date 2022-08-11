from ospark.nn.layers.feed_forward import *
from typing import NoReturn, Union
import tensorflow as tf


class DeepFeedForwardLayer(FeedForwardLayer):

    def __init__(self,
                 obj_name: str,
                 embedding_size: int,
                 scale_rate: int,
                 dropout_rate: float,
                 is_training: Optional[bool]=None,
                 activation: Optional[Activation]=None) -> NoReturn:
        super(DeepFeedForwardLayer, self).__init__(obj_name=obj_name,
                                                   embedding_size=embedding_size,
                                                   scale_rate=scale_rate,
                                                   activation=activation,
                                                   dropout_rate=dropout_rate,
                                                   is_training=is_training)
        self._res_adaptive_value = 1

    @property
    def res_adaptive_value(self) -> Union[int, tf.Tensor]:
        return self._res_adaptive_value

    def profiling_phase(self, input_data: tf.Tensor, catch_variance: list):
        self._res_adaptive_value = 1.

        output   = self.__call__(input_data)
        _, variance = tf.nn.moments(output, axes=[0, 1])
        catch_variance.append(variance)

        self._res_adaptive_value = tf.sqrt(tf.reduce_sum(catch_variance, axis=0))[tf.newaxis, tf.newaxis, :]
        return output, catch_variance

    def residual_net(self, input_data: tf.Tensor) -> tf.Tensor:
        output = tf.multiply(input_data, self.res_adaptive_value)
        return output

    def back_to_standard(self) -> NoReturn:
        self._res_adaptive_value = 1.