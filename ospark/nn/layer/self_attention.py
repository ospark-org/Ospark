
import tensorflow as tf
from . import Layer
from ospark.nn.component.normalization import Normalization
from typing import NoReturn, Tuple, Callable, Optional
import ospark

class SelfAttention(Layer):

    def __init__(self, 
                 obj_name: str, 
                 embedding_size: int, 
                 head_number: int, 
                 normalization: Optional[Normalization]=None,
                 look_ahead: bool=False) -> NoReturn:
        super().__init__(obj_name=obj_name)
        self._embedding_size            = embedding_size
        self._head_number               = head_number
        self._look_ahead                = look_ahead
        self._sequence_length           = None
        self._normalization             = normalization or ospark.normalization.LayerNormalization()

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def head_number(self) -> int:
        return self._head_number

    @property
    def normalization(self) -> Normalization:
        return self._normalization

    @property
    def look_ahead(self) -> bool:
        return self._look_ahead

    @property
    def sequence_length(self) -> None:
        return self._sequence_length

    @property
    def look_ahead_mask(self) -> tf.Tensor:
        return 1 - tf.linalg.band_part(tf.ones((self.sequence_length, self.sequence_length)), -1, 0)

    def initialize(self) -> NoReturn:
        self.assign(ospark.weight.truncated_normal(
                                obj_name="QKV_weights",
                                weight_shape=[3, self.head_number, self.embedding_size, int(self.embedding_size / self.head_number)]
        ))
        self.assign(ospark.weight.truncated_normal(
                                obj_name="output_weights",
                                weight_shape=[self.embedding_size, self.embedding_size]
        ))

    def model(self, input_data: tf.Tensor, mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        Q, K, V = self.QKV_process(input_data)
        main_output = self.attention_layer(Q, K, V, mask)
        residual_output = self.residual_net(input_data)
        added_residual = tf.add(main_output, residual_output)
        layer_output = self.normalization(added_residual)
        return layer_output

    def QKV_process(self, input_data: tf.Tensor) -> Tuple[tf.Tensor]:
        input_data = tf.tile(input_data[:, tf.newaxis, tf.newaxis, :, :], [1, 3, 4, 1, 1])
        QKV = tf.matmul(input_data, self.assigned.QKV_weights)
        Q, K, V = tf.unstack(QKV, num=3, axis=1)
        return Q, K, V

    def attention_layer(self, 
                        Q: tf.Tensor, 
                        K: tf.Tensor, 
                        V: tf.Tensor, 
                        mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        attention_value = self.attention(Q, K, V, mask)
        layer_output = tf.matmul(attention_value, self.assigned.output_weights)
        return layer_output

    def attention(self, Q: tf.Tensor, K: tf.Tensor, V: tf.Tensor, mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        K = tf.transpose(K, [0, 1, 3, 2])
        scaled_dot_product = tf.matmul(Q, K) / tf.math.sqrt(tf.cast(self.embedding_size, dtype=tf.float32))
        if self.look_ahead:
            self._sequence_length = tf.shape(Q)[-2]
            scaled_dot_product += (self.look_ahead_mask * -1e9)
        if mask is not None:
            scaled_dot_product += (mask * -1e9)
        scaled_dot_product = tf.nn.softmax(scaled_dot_product)
        output = tf.matmul(scaled_dot_product, V)
        concat_output = tf.concat(tf.unstack(output, self.head_number, axis=1), axis=2)
        return concat_output

    def residual_net(self, input_data: tf.Tensor) -> tf.Tensor:
        return input_data

    def __call__(self, mask: Optional[tf.Tensor]=None) -> Callable[[tf.Tensor], tf.Tensor]:
        def model(input_data: tf.Tensor) -> tf.Tensor:
            return self.model(input_data, mask)
        return model

class EncoderDecoderAttention(SelfAttention):

    def __init__(self,
                 obj_name: str, 
                 embedding_size: int, 
                 head_number: int, 
                 normalization: Optional[Normalization]=None,
                 look_ahead: bool=False) -> NoReturn:
        super().__init__(obj_name=obj_name, 
                         embedding_size=embedding_size, 
                         head_number=head_number, 
                         normalization=normalization,
                         look_ahead=look_ahead)
        self._encoder_output = None

    @property
    def encoder_output(self) -> None:
        return self._encoder_output

    def initialize(self) -> NoReturn:
        self.assign(ospark.weight.truncated_normal(
                                obj_name="Q_weights",
                                weight_shape=[1, self.head_number, self.embedding_size, int(self.embedding_size / self.head_number)]
        ))
        self.assign(ospark.weight.truncated_normal(
                                obj_name="KV_weights",
                                weight_shape=[2, self.head_number, self.embedding_size, int(self.embedding_size / self.head_number)]
        ))
        self.assign(ospark.weight.truncated_normal(
                                obj_name="output_weights",
                                weight_shape=[self.embedding_size, self.embedding_size]
        ))

    def QKV_process(self, input_data: tf.Tensor) -> Tuple[tf.Tensor]:
        encoder_output = tf.tile(self.encoder_output[:, tf.newaxis, tf.newaxis, :, :], [1, 2, 4, 1, 1])
        input_data = tf.tile(input_data[:, tf.newaxis, :, :], [1, 4, 1, 1])
        Q    = tf.matmul(input_data, self.assigned.Q_weights)
        KV   = tf.matmul(encoder_output, self.assigned.KV_weights)
        K, V = tf.unstack(KV, num=2, axis=1)
        return Q, K, V

    def __call__(self, mask: tf.Tensor, encoder_output: tf.Tensor) -> Callable[[tf.Tensor], tf.Tensor]:
        mask = tf.cast(tf.math.equal(encoder_output[:, :, 0], 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
        def model(input_data: tf.Tensor) -> tf.Tensor:
            self._encoder_output = encoder_output
            return self.model(input_data, mask)
        return model