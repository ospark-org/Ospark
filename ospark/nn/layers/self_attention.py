import tensorflow as tf

import ospark.utility.weight_initializer
from ospark.nn.layers import Layer
from ospark.nn.layers.normalization import Normalization
from typing import *
import ospark
import time


class SelfAttentionLayer(Layer):

    def __init__(self, 
                 obj_name: str, 
                 embedding_size: int, 
                 head_number: int,
                 dropout_rate: float,
                 is_training: Optional[bool]=None,
                 use_look_ahead: Optional[bool]=False) -> NoReturn:
        super().__init__(obj_name=obj_name, is_training=is_training)
        assert embedding_size % head_number == 0

        self._depth           = embedding_size // head_number
        self._embedding_size  = embedding_size
        self._head_number     = head_number
        self._sequence_length = None
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

    def in_creating(self) -> NoReturn:
        self._q_weights = ospark.utility.weight_initializer.glorot_uniform(
            obj_name="q_weights",
            shape=[self.embedding_size, self.embedding_size]
        )
        self._k_weights = ospark.utility.weight_initializer.glorot_uniform(
            obj_name="k_weights",
            shape=[self.embedding_size, self.embedding_size]
        )
        self._v_weights = ospark.utility.weight_initializer.glorot_uniform(
            obj_name="v_weights",
            shape=[self.embedding_size, self.embedding_size]
        )
        self._q_bias = ospark.utility.weight_initializer.zeros(
            obj_name="q_bias",
            shape=[self.embedding_size]
        )
        self._k_bias = ospark.utility.weight_initializer.zeros(
            obj_name="k_bias",
            shape=[self.embedding_size]
        )
        self._v_bias = ospark.utility.weight_initializer.zeros(
            obj_name="v_bias",
            shape=[self.embedding_size]
        )
        self._output_weights = ospark.utility.weight_initializer.glorot_uniform(
            obj_name="output_weights",
            shape=[self.embedding_size, self.embedding_size]
        )
        self._output_bias = ospark.utility.weight_initializer.zeros(
            obj_name="output_bias",
            shape=[self.embedding_size]
        )
        self._norm = ospark.nn.layers.normalization.LayerNormalization(obj_name="layer_norm",
                                                                       layer_dimension=self.embedding_size)

    def pipeline(self, input_data: tf.Tensor, mask: Optional[tf.Tensor]=None):
        return self.layer_calculation(Q_input=input_data,
                                      K_input=input_data,
                                      V_input=input_data,
                                      mask=mask)

    def layer_calculation(self,
                          Q_input: tf.Tensor,
                          K_input: tf.Tensor,
                          V_input: tf.Tensor,
                          mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        batch_size = tf.shape(Q_input)[0]

        Q, K, V     = self.QKV_process(Q_input=Q_input, K_input=K_input, V_input=V_input, batch_size=batch_size)
        main_output = self.attention_layer(Q=Q, K=K, V=V, batch_size=batch_size, mask=mask)

        residual_output = self.residual_net(input_data=Q_input)
        added_residual  = tf.add(self.dropout_layer(main_output, training=self.is_training), residual_output)
        layer_output    = self._norm(added_residual)
        return layer_output

    def QKV_process(self,
                    Q_input: tf.Tensor,
                    K_input: tf.Tensor,
                    V_input: tf.Tensor,
                    batch_size: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        Q = tf.matmul(Q_input, self._q_weights) + self._q_bias  # [batch, seq, d_model]
        K = tf.matmul(K_input, self._k_weights) + self._k_bias
        V = tf.matmul(V_input, self._v_weights) + self._v_bias
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
        layer_output    = tf.matmul(attention_value, self._output_weights) + self._output_bias
        return layer_output

    def attention(self,
                  Q: tf.Tensor,
                  K: tf.Tensor,
                  V: tf.Tensor,
                  batch_size: tf.Tensor,
                  mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        K = tf.transpose(K, [0, 1, 3, 2])  # BHLD -> BHDL
        scaled_dot_product = tf.matmul(Q, K) / tf.math.sqrt(tf.cast(self.embedding_size, dtype=tf.float32))  # BHLD * BHDL -> BHLL
        if self.use_look_ahead and mask is not None:
            self._sequence_length = tf.shape(Q)[-2]
            scaled_dot_product += (tf.cast(tf.math.not_equal(mask + self.look_ahead_mask, 0), tf.float32) * -1e9)
        elif mask is not None:
            scaled_dot_product += (mask * -1e9)
        scaled_dot_product = tf.nn.softmax(scaled_dot_product, axis=-1)
        scaled_attention   = tf.matmul(scaled_dot_product, V) # BHLL * BHLD -> BHLD
        scaled_attention   = tf.transpose(scaled_attention, [0, 2, 1, 3]) # BHLD -> BLHD
        concat_output      = tf.reshape(scaled_attention, [batch_size, -1, self.embedding_size])
        return concat_output

    def residual_net(self, input_data: tf.Tensor) -> tf.Tensor:
        return input_data


class EncoderDecoderAttentionLayer(SelfAttentionLayer):

    def __init__(self,
                 obj_name: str, 
                 embedding_size: int, 
                 head_number: int,
                 dropout_rate: float,
                 is_training: Optional[bool]=False,
                 use_look_ahead: Optional[bool]=False) -> NoReturn:
        super().__init__(obj_name=obj_name, 
                         embedding_size=embedding_size, 
                         head_number=head_number,
                         dropout_rate=dropout_rate,
                         is_training=is_training,
                         use_look_ahead=use_look_ahead)

    def pipeline(self, input_data: tf.Tensor, encoder_output: tf.Tensor, mask: Optional[tf.Tensor]=None):
        return self.layer_calculation(Q_input=input_data,
                                      K_input=encoder_output,
                                      V_input=encoder_output,
                                      mask=mask)


class KernelFunction:

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class SoftmaxKernel(KernelFunction):

    def __call__(self, input_data: tf.Tensor, projection_matrix: tf.Tensor):
        normalizer = 1.0 / tf.math.sqrt(tf.cast(input_data.shape[-1], dtype=tf.float32))
        input_data = input_data * normalizer
        scaling    = 1.0 / tf.math.sqrt(tf.cast(projection_matrix.shape[0], dtype=tf.float32))

        mapping_result  = tf.matmul(input_data, tf.transpose(projection_matrix, [1, 0])) # blhd * dm -> blhm
        correction_part = tf.square(input_data)
        correction_part = tf.math.reduce_sum(correction_part, axis=-1, keepdims=True) / 2.0 # blh1
        result = scaling * tf.math.exp(mapping_result - correction_part)
        return result


@tf.custom_gradient
def causal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR causal attention A_{masked}V.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].
  Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
  """

  result = []
  sums = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

  for index in range(qs.shape[0]):
    sums = sums + tf.einsum("ijk,ijl->ijkl", ks[index], vs[index]) # BHMD
    result.append(tf.einsum("ijkl,ijk->ijl", sums, qs[index])[None, Ellipsis])

  result = tf.concat(result, axis=0)

  def grad(res_grad):

    grads = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

    gr_sums = sums

    q_grads = []
    k_grads = []
    v_grads = []

    for index in range(qs.shape[0] - 1, -1, -1):

      q_grads.append(
          tf.einsum("ijkl,ijl->ijk", gr_sums, res_grad[index])[None, Ellipsis])
      grads = grads + tf.einsum("ijk,ijl->ijkl", qs[index], res_grad[index])
      k_grads.append(tf.einsum("ijkl,ijl->ijk", grads, vs[index])[None, Ellipsis])
      v_grads.append(tf.einsum("ijkl,ijk->ijl", grads, ks[index])[None, Ellipsis])
      gr_sums = gr_sums - tf.einsum("ijk,ijl->ijkl", ks[index], vs[index])

    q_grads = tf.concat(q_grads[::-1], axis=0)
    k_grads = tf.concat(k_grads[::-1], axis=0)
    v_grads = tf.concat(v_grads[::-1], axis=0)

    return q_grads, k_grads, v_grads

  return result, grad


@tf.custom_gradient
def causal_denominator(qs, ks):
  """Computes FAVOR normalizer in causal attention.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
  Returns:
    FAVOR normalizer in causal attention.
  """

  result = []
  sums = tf.zeros_like(ks[0])

  for index in range(qs.shape[0]):
    sums = sums + ks[index]
    result.append(tf.reduce_sum(qs[index] * sums, axis=2)[None, Ellipsis])
    # result.append(tf.einsum("ijk,ijk->ij", qs[index], sums)[None, Ellipsis])

  result = tf.concat(result, axis=0)

  def grad(res_grad):

    k_grad = tf.zeros_like(ks[0])

    gr_sums = sums

    q_grads = []
    k_grads = []

    for index in range(qs.shape[0] - 1, -1, -1):

      q_grads.append(
          tf.einsum("ijk,ij->ijk", gr_sums, res_grad[index])[None, Ellipsis])
      k_grad = k_grad + tf.einsum("ijk,ij->ijk", qs[index], res_grad[index])
      k_grads.append(k_grad[None, Ellipsis])
      gr_sums = gr_sums - ks[index]

    q_grads = tf.concat(q_grads[::-1], axis=0)
    k_grads = tf.concat(k_grads[::-1], axis=0)

    return q_grads, k_grads

  return result, grad


class FavorAttentionLayer(SelfAttentionLayer):

    def __init__(self,
                 obj_name: str,
                 embedding_size: int,
                 head_number: int,
                 dropout_rate: float,
                 random_projections_number: int,
                 kernel_function: Optional[KernelFunction]=None,
                 is_training: Optional[bool]=False,
                 use_look_ahead: Optional[bool]=False) -> NoReturn:
        super().__init__(obj_name=obj_name,
                         is_training=is_training,
                         embedding_size=embedding_size,
                         head_number=head_number,
                         dropout_rate=dropout_rate,
                         use_look_ahead=use_look_ahead)
        self._random_projections_number = random_projections_number
        self._inner_dimension           = int(embedding_size / head_number)
        self._kernel_function           = kernel_function or SoftmaxKernel()

    @property
    def random_projections_number(self) -> int:
        return self._random_projections_number

    @property
    def inner_dimension(self) -> int:
        return self._inner_dimension

    @property
    def kernel_function(self) -> KernelFunction:
        return self._kernel_function

    def create_projections_matrix(self,
                                  random_projections_number: int,
                                  embedding_size: int,
                                  seed: Optional[int]=0):
        blocks = []

        block_number = int(random_projections_number / embedding_size)
        current_seed = seed

        for _ in range(block_number):
            orthogonal_matrix = self.get_orthogonal_matrix(matrix_shape=[embedding_size, embedding_size],
                                                           seed=current_seed)
            blocks.append(orthogonal_matrix)
            current_seed = current_seed + 1

        remaining_row = random_projections_number - block_number * embedding_size
        if remaining_row > 0:
            orthogonal_matrix = self.get_orthogonal_matrix(matrix_shape=[embedding_size, embedding_size],
                                                           seed=current_seed)
            blocks.append(orthogonal_matrix[:remaining_row])

        final_matrix = tf.experimental.numpy.vstack(blocks)
        multiplier   = tf.sqrt(float(embedding_size)) * tf.ones(shape=[random_projections_number])
        return tf.linalg.matmul(tf.linalg.diag(multiplier), final_matrix)

    def get_orthogonal_matrix(self, matrix_shape: List[int], seed: int) -> tf.Tensor:
        tf.random.set_seed(seed=seed)
        random_feature = tf.random.normal(shape=matrix_shape)
        q, _ = tf.linalg.qr(random_feature)
        return q

    def attention(self,
                  Q: tf.Tensor,
                  K: tf.Tensor,
                  V: tf.Tensor,
                  batch_size: tf.Tensor,
                  mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        seed = tf.cast(tf.math.ceil(tf.math.abs(tf.reduce_sum(Q) * 1e8)), tf.int32)
        projection_matrix = self.create_projections_matrix(random_projections_number=self.random_projections_number,
                                                           embedding_size=self.inner_dimension,
                                                           seed=seed)
        Q = tf.transpose(Q, [2, 0, 1, 3])  # BHLD -> LBHD
        K = tf.transpose(K, [2, 0, 1, 3])  # BHLD -> LBHD
        V = tf.transpose(V, [2, 0, 1, 3])  # BHLD -> LBHD
        Q = self.kernel_function(input_data=Q, projection_matrix=projection_matrix)  # LBHD -> LBHM
        K = self.kernel_function(input_data=K, projection_matrix=projection_matrix)  # LBHD -> LBHM
        if self.use_look_ahead:
            normalizer       = causal_numerator(qs=Q, ks=K, vs=V) # LBHD
            denominator      = causal_denominator(qs=Q, ks=K) # LBH
            attention_output = normalizer / denominator[..., tf.newaxis] # LBHD / LBH1
            attention_output = tf.transpose(attention_output, [1, 0, 2, 3]) # LBHD -> BLHD
        else:
            normalizer       = self.calculate_numerator(qs=Q, ks=K, vs=V)  # LBHD
            denominator      = self.calculate_denominator(qs=Q, ks=K)[..., tf.newaxis]  # LBH -> LBH1
            attention_output = tf.transpose(normalizer / denominator, [1, 0, 2, 3]) # LBHD -> BLHD
        attention_output     = tf.reshape(attention_output,
                                          shape=[batch_size, -1, self.embedding_size])
        return attention_output

    def calculate_numerator(self, qs: tf.Tensor, ks: tf.Tensor, vs: tf.Tensor) -> tf.Tensor:
        kvs = tf.einsum("lbhm,lbhd->bhmd", ks, vs)
        return tf.einsum("lbhm,bhmd->lbhd", qs, kvs)

    def calculate_denominator(self, qs: tf.Tensor, ks: tf.Tensor) -> tf.Tensor:
        # qs -> LBHM, ks -> LBHM
        all_ones = tf.ones([ks.shape[0]])
        ks_sum = tf.einsum("lbhm,l->bhm", ks, all_ones)
        return tf.einsum("lbhm,bhm->lbh", qs, ks_sum)

    def bidirectional_attention(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor):
        B, L, H, D = tf.shape(v)
        c       = tf.concat([v, tf.ones(shape=[B, L, H, 1])], axis=-1) # BLHD -> BLH D+1
        # k       = tf.transpose(k, [0, 1, 3, 2]) # BLHM -> BLMH
        buf1    = tf.matmul(k, c, transpose_a=True) # BLMH * BLHD+1 -> BLMD+1
        buf2    = tf.matmul(q, buf1) # BLHM * BLMD+1 -> BLHD+1
        buf3    = buf2[:, :, :, :D] # BLHD
        buf4    = buf2[:, :, :, D:] # BLH1
        return buf3 / tf.reduce_sum(buf4, axis=1, keepdims=True) # BLHD

    def unidirectional_attention(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor):
        B, L, H, D = tf.shape(v)

        c = tf.concat([v, tf.ones(shape=[B, L, H, 1])], axis=-1) # BLHD -> BLH D+1

        G = tf.matmul(k[..., tf.newaxis], tf.reshape(c, shape=[B, L, H, 1, D + 1])) # BLHM1 * BLH1 D+1 -> BLHM D+1


        mask = tf.reshape(1 - tf.linalg.band_part(tf.ones((L, L)), -1, 0), shape=[1, L, L , 1, 1, 1])
        mask = tf.tile(mask, [B, 1, 1, H, self.random_projections_number, D+1]) # BLLHM D+1
        s = time.time()
        Gps = tf.linalg.einsum("ijklmn, ijlmn->ijlmn", mask, G)
        print("1",time.time() - s)
        s = time.time()
        Gps = tf.concat([tf.reduce_sum(G[:, :i, ...], axis=1, keepdims=True)for i in range(L)], axis=1) # BLHM D+1
        print("2",time.time() - s)

        q = tf.reshape(q, shape=[B, L, H, 1, -1]) # BLH1M

        buf2 = tf.squeeze(tf.matmul(q, Gps))  # BLH1M * BLHM D+1 -> BLH1D+1 -> BLHD+1


        # buf2 = tf.concat([tf.reshape(tf.matmul(tf.reshape(q[:, i, ...], shape=[B, H, 1, -1]), tf.reduce_sum(G[:, :i, ...], axis=1)), [B, 1, H, D + 1])
        #        for i in range(L)], axis=1) # B1H1M * B1HM D+1 -> B1H1 D+1 for L -> BLH D+1

        buf3 = buf2[:, :, :, :D] # BLHD
        buf4 = buf2[:, :, :, D:] # BLH1
        return buf3 / tf.reduce_sum(buf4, axis=1, keepdims=True) # BLHD


class EncodeDecodeFavorAttention(FavorAttentionLayer):

    def __init__(self,
                 obj_name: str,
                 embedding_size: int,
                 head_number: int,
                 dropout_rate: float,
                 random_projections_number: int,
                 kernel_function: Optional[KernelFunction]=None,
                 is_training: Optional[bool]=False,
                 use_look_ahead: Optional[bool]=False) -> NoReturn:
        super(EncodeDecodeFavorAttention, self).__init__(obj_name=obj_name,
                                                         embedding_size=embedding_size,
                                                         head_number=head_number,
                                                         dropout_rate=dropout_rate,
                                                         random_projections_number=random_projections_number,
                                                         kernel_function=kernel_function,
                                                         is_training=is_training,
                                                         use_look_ahead=use_look_ahead)

    def pipeline(self, input_data: tf.Tensor, encoder_output: tf.Tensor, mask: Optional[tf.Tensor]=None):
        return self.layer_calculation(Q_input=input_data,
                                      K_input=encoder_output,
                                      V_input=encoder_output,
                                      mask=mask)