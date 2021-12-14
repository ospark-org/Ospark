
import tensorflow as tf
from . import Layer
from ospark.nn.component.normalization import Normalization
from typing import NoReturn, Tuple, Callable, Optional
import ospark

class SelfAttentionLayer(Layer):

    def __init__(self, 
                 obj_name: str, 
                 embedding_size: int, 
                 head_number: int,
                 dropout_rate: float,
                 is_training: Optional[bool]=False,
                 use_look_ahead: Optional[bool]=False,
                 normalization: Optional[Normalization]=None) -> NoReturn:
        super().__init__(obj_name=obj_name, is_training=is_training)
        assert embedding_size % head_number == 0

        self._depth           = embedding_size // head_number
        self._embedding_size  = embedding_size
        self._head_number     = head_number
        self._sequence_length = None
        self._normalization   = normalization or ospark.normalization.LayerNormalization(layer_dimension=embedding_size)
        self._dropout_layer   = tf.keras.layers.Dropout(rate=dropout_rate)
        self._use_look_ahead  = use_look_ahead

    @property
    def depth(self) -> int:
        return self._depth

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
    def sequence_length(self) -> None:
        return self._sequence_length

    @property
    def dropout_layer(self) -> tf.keras.layers.Dropout:
        return self._dropout_layer

    @property
    def use_look_ahead(self) -> bool:
        return self._use_look_ahead

    @property
    def look_ahead_mask(self) -> tf.Tensor:
        return 1 - tf.linalg.band_part(tf.ones((self.sequence_length, self.sequence_length)), -1, 0)

    def on_creating(self) -> NoReturn:
        self.assign(ospark.weight.glorot_uniform(
            obj_name="Q_weights",
            weight_shape=[self.embedding_size, self.embedding_size]
        ))
        self.assign(ospark.weight.glorot_uniform(
            obj_name="K_weights",
            weight_shape=[self.embedding_size, self.embedding_size]
        ))
        self.assign(ospark.weight.glorot_uniform(
            obj_name="V_weights",
            weight_shape=[self.embedding_size, self.embedding_size]
        ))
        self.assign(ospark.weight.glorot_uniform(
            obj_name="output_weights",
            weight_shape=[self.embedding_size, self.embedding_size]
        ))
        self.assign(ospark.weight.zeros(
            obj_name="Q_bias",
            weight_shape=[self.embedding_size]
        ))
        self.assign(ospark.weight.zeros(
            obj_name="K_bias",
            weight_shape=[self.embedding_size]
        ))
        self.assign(ospark.weight.zeros(
            obj_name="V_bias",
            weight_shape=[self.embedding_size]
        ))
        self.assign(ospark.weight.zeros(
            obj_name="output_bias",
            weight_shape=[self.embedding_size]
        ))
        self.assign(component=self.normalization, name="norm")

    def model(self, input_data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        batch_size = tf.shape(input_data[0])[0]
        Q_input, K_input, V_input = input_data
        Q, K, V = self.QKV_process(Q_input=Q_input, K_input=K_input, V_input=V_input, batch_size=batch_size)
        main_output = self.attention_layer(Q=Q, K=K, V=V, batch_size=batch_size, mask=mask)
        residual_output = self.residual_net(input_data=Q_input)
        added_residual = tf.add(self.dropout_layer(main_output, training=self.is_training), residual_output)
        layer_output = self.assigned.norm(added_residual)
        return layer_output

    def QKV_process(self,
                    Q_input: tf.Tensor,
                    K_input: tf.Tensor,
                    V_input: tf.Tensor,
                    batch_size: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        Q = tf.matmul(Q_input, self.assigned.Q_weights) + self.assigned.Q_bias  # [batch, seq, d_model]
        K = tf.matmul(K_input, self.assigned.K_weights) + self.assigned.K_bias
        V = tf.matmul(V_input, self.assigned.V_weights) + self.assigned.V_bias
        Q = self.split_head(input_data=Q, batch_size=batch_size) # [batch, head_number, seq, depth]
        K = self.split_head(input_data=K, batch_size=batch_size)
        V = self.split_head(input_data=V, batch_size=batch_size)
        return Q, K, V

    def split_head(self, input_data: tf.Tensor, batch_size: tf.int32) -> tf.Tensor:
        split_result     = tf.reshape(input_data, [batch_size, -1, self.head_number, self.depth]) # [batch, seq, head_number, depth]
        transpose_result = tf.transpose(split_result, [0, 2, 1, 3]) # [batch, head_number, seq, depth]
        return transpose_result

    def attention_layer(self, 
                        Q: tf.Tensor, 
                        K: tf.Tensor, 
                        V: tf.Tensor,
                        batch_size: tf.Tensor,
                        mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        attention_value = self.attention(Q=Q, K=K, V=V, batch_size=batch_size, mask=mask)
        layer_output = tf.matmul(attention_value, self.assigned.output_weights) + self.assigned.output_bias
        return layer_output

    def attention(self,
                  Q: tf.Tensor,
                  K: tf.Tensor,
                  V: tf.Tensor,
                  batch_size: tf.Tensor,
                  mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        K = tf.transpose(K, [0, 1, 3, 2])
        scaled_dot_product = tf.matmul(Q, K) / tf.math.sqrt(tf.cast(self.embedding_size, dtype=tf.float32))
        if self.use_look_ahead and mask is not None:
            self._sequence_length = tf.shape(Q)[-2]
            scaled_dot_product   += (tf.cast(tf.math.not_equal(mask + self.look_ahead_mask, 0), tf.float32) * -1e9)
        elif mask is not None:
            scaled_dot_product += (mask * -1e9)
        scaled_dot_product = tf.nn.softmax(scaled_dot_product, axis=-1)
        scaled_attention   = tf.matmul(scaled_dot_product, V)
        scaled_attention   = tf.transpose(scaled_attention, [0, 2, 1, 3]) # [batch, seq, head_number, d_model]
        concat_output      = tf.reshape(scaled_attention, [batch_size, -1, self.embedding_size])
        return concat_output

    def residual_net(self, input_data: tf.Tensor) -> tf.Tensor:
        return input_data

    def __call__(self, mask: Optional[tf.Tensor]=None) -> Callable[[tf.Tensor], tf.Tensor]:
        def model(input_data: tf.Tensor) -> tf.Tensor:
            return self.model((input_data, input_data, input_data), mask)
        return model

class EncoderDecoderAttentionLayer(SelfAttentionLayer):

    def __init__(self,
                 obj_name: str, 
                 embedding_size: int, 
                 head_number: int,
                 dropout_rate: float,
                 is_training: Optional[bool]=False,
                 use_look_ahead: Optional[bool]=False,
                 normalization: Optional[Normalization]=None) -> NoReturn:
        super().__init__(obj_name=obj_name, 
                         embedding_size=embedding_size, 
                         head_number=head_number, 
                         normalization=normalization,
                         dropout_rate=dropout_rate,
                         is_training=is_training,
                         use_look_ahead=use_look_ahead)

    def __call__(self, mask: tf.Tensor, encoder_output: tf.Tensor) -> Callable[[tf.Tensor], tf.Tensor]:
        def model(input_data: tf.Tensor) -> tf.Tensor:
            return self.model((input_data, encoder_output, encoder_output), mask)
        return model