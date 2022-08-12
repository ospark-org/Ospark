import ospark.utility.weight_initializer
from . import Layer
from ospark.nn.layers.normalization import Normalization
from ospark.nn.component.activation import Activation
from typing import NoReturn, Optional
import tensorflow as tf 
import ospark


class FeedForwardLayer(Layer):

    def __init__(self, 
                 obj_name: str, 
                 embedding_size: int, 
                 scale_rate: int,
                 dropout_rate: float,
                 is_training: Optional[bool]=None,
                 activation: Optional[Activation]=None) -> NoReturn:
        super().__init__(obj_name=obj_name, is_training=is_training)
        self._activation     = activation or ospark.activation.ReLU()
        self._embedding_size = embedding_size
        self._scale_rate     = scale_rate
        self._dropout_layer  = tf.keras.layers.Dropout(rate=dropout_rate)

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def scale_rate(self) -> int:
        return self._scale_rate

    @property
    def activation(self) -> Activation:
        return self._activation

    @property
    def dropout_layer(self) -> tf.keras.layers.Dropout:
        return self._dropout_layer

    def in_creating(self) -> NoReturn:
        self._mapping2high_dimensional = ospark.utility.weight_initializer.glorot_uniform(
                                obj_name="mapping2high_dimensional",
                                shape=[self.embedding_size, self.scale_rate * self.embedding_size])
        self._mapping2low_dimensional = ospark.utility.weight_initializer.glorot_uniform(
                                obj_name="mapping2low_dimensional",
                                shape=[self.scale_rate * self.embedding_size, self.embedding_size])
        self._high_dimensional_bias = ospark.utility.weight_initializer.zeros(
                                obj_name="high_dimensional_bias",
                                shape=[self.scale_rate * self.embedding_size])
        self._low_dimensional_bias  = ospark.utility.weight_initializer.zeros(
                                obj_name="low_dimensional_bias",
                                shape=[self.embedding_size])
        self._norm = ospark.nn.layers.normalization.LayerNormalization(obj_name="layer_norm",
                                                                       layer_dimension=self.embedding_size)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        main_output          = self.feedforward(input_data)
        residual_output      = self.residual_net(input_data)
        added_residual       = tf.add(self.dropout_layer(main_output, training=self.is_training), residual_output)
        normalization_output = self._norm(added_residual)
        return normalization_output

    def feedforward(self, input_data: tf.Tensor) -> tf.Tensor:
        mapping2high_dimensional = tf.matmul(input_data, self._mapping2high_dimensional) + self._high_dimensional_bias
        activated_outputs        = self.activation(mapping2high_dimensional)
        mapping2low_dimensional  = tf.matmul(activated_outputs, self._mapping2low_dimensional) + self._low_dimensional_bias
        return mapping2low_dimensional

    def residual_net(self, input_data: tf.Tensor) -> tf.Tensor:
        return input_data
