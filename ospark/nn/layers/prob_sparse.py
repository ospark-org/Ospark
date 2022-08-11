from ospark.nn.layers.self_attention import SelfAttentionLayer
import tensorflow as tf
import numpy as np
from ospark.nn.layers.normalization import Normalization
from typing import NoReturn, Tuple, Optional


class ProbSparseAttentionLayer(SelfAttentionLayer):

    def __init__(self,
                 obj_name: str,
                 embedding_size: int,
                 head_number: int,
                 sample_factor: float,
                 dropout_rate: float,
                 is_training: Optional[bool]=None,
                 use_look_ahead: Optional[bool]=False) -> NoReturn:
        super().__init__(obj_name=obj_name,
                         embedding_size=embedding_size,
                         head_number=head_number,
                         dropout_rate=dropout_rate,
                         is_training=is_training,
                         use_look_ahead=use_look_ahead)
        self._sample_factor = sample_factor
        self._top_u         = None

    @property
    def sample_factor(self) -> float:
        return self._sample_factor

    @property
    def top_u(self) -> None:
        return self._top_u
    def QKV_process(self,
                    Q_input: tf.Tensor,
                    K_input: tf.Tensor,
                    V_input: tf.Tensor,
                    batch_size: tf.int32) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        Q, K, V = super().QKV_process(Q_input=Q_input, K_input=K_input, V_input=V_input)
        Q_bar = self.sampling(Q, K)
        return Q_bar, K, V

    def sampling(self, Q: tf.Tensor, K: tf.Tensor) -> tf.Tensor:
        batch, head_number, Q_sequence_length, embedding_size = tf.shape(Q)
        _    ,           _, K_sequence_length,              _ = tf.cast(tf.shape(K), dtype=tf.float32)
        self._sequence_length = Q_sequence_length
        sampling_number_U     = tf.cast(tf.math.ceil(self.sample_factor * tf.math.log(K_sequence_length)), dtype=tf.int32)
        sampling_number_u     = tf.cast(tf.math.ceil(self.sample_factor * tf.math.log(tf.cast(Q_sequence_length, dtype=tf.float32))), dtype=tf.int32)
        K_samples             = tf.random.categorical(tf.ones(shape=[1, K_sequence_length]), sampling_number_U)
        K_bar                 = tf.gather(K, K_samples[0], axis=-2)
        sample_score          = tf.matmul(Q, tf.transpose(K_bar, [0, 1, 3, 2]))
        max_mean_measurement  = tf.reduce_max(sample_score, axis=-1) - tf.reduce_mean(sample_score, axis=-1)
        self._top_u           = tf.nn.top_k(max_mean_measurement, sampling_number_u, sorted=False).indices.numpy()
        Q_bar = Q.numpy()[np.arange(batch)[:, None, None],
                          np.arange(head_number)[None, :, None],
                          self.top_u,
                          :]
        return Q_bar

    def attention(self, Q: tf.Tensor, K: tf.Tensor, V: tf.Tensor, mask: tf.Tensor=None) -> tf.Tensor:
        batch, head_number, V_sequence_length, embedding_size = tf.shape(V)
        K = tf.transpose(K, [0, 1, 3, 2])
        scaled_dot_product = tf.matmul(Q, K) / tf.math.sqrt(tf.cast(self.embedding_size, dtype=tf.float32))
        if self.use_look_ahead and mask is not None:
            scaled_dot_product += (tf.cast(tf.math.not_equal(self.look_ahead_mask[self.top_u, :] +
                                                             mask[np.arange(batch)[:, None, None],
                                                                  np.arange(head_number)[None, :, None],
                                                                  self.top_u, :], 0), tf.float32) * -1e9)
        elif mask is not None:
            scaled_dot_product += (mask[np.arange(batch)[:, None, None],
                                        np.arange(head_number)[None, :, None],
                                        self.top_u, :] * -1e9)
        mean_V = self.create_mean_value(V)
        scaled_dot_product = tf.nn.softmax(scaled_dot_product)
        mean_V.numpy()[np.arange(batch)[:, None, None],
                       np.arange(head_number)[None, :, None],
                       self.top_u, :] = tf.matmul(scaled_dot_product, V).numpy()
        concat_output = tf.concat(tf.unstack(mean_V, self.head_number, axis=1), axis=2)
        return concat_output

    def create_mean_value(self, value: tf.Tensor) -> tf.Tensor:
        return tf.tile(tf.reduce_mean(value, axis=-2, keepdims=True), [1, 1, self.sequence_length, 1])