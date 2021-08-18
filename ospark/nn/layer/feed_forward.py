
from . import Layer
from ospark.nn.component.normalization import Normalization
from ospark.nn.component.activation import Activation
from typing import NoReturn, Optional
import tensorflow as tf 
import ospark


class FeedForward(Layer):

    def __init__(self, 
                 obj_name: str, 
                 embedding_size: int, 
                 scale_rate: int, 
                 activation: Optional[Activation]=None,
                 normalization: Optional[Normalization]=None) -> NoReturn:
        super().__init__(obj_name=obj_name)
        self._normalization  = normalization or ospark.normalization.LayerNormalization()
        self._activation     = activation or ospark.activation.ReLU()
        self._embedding_size = embedding_size
        self._scale_rate     = scale_rate

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
    def normalization(self) -> Normalization:
        return self._normalization

    def initialize(self) -> NoReturn:
        self.assign(ospark.weight.truncated_normal(
                                obj_name="mapping2high_dimensional", 
                                weight_shape=[self.embedding_size, self.scale_rate * self.embedding_size]))
        self.assign(ospark.weight.truncated_normal(
                                obj_name="mapping2low_dimensional", 
                                weight_shape=[self.scale_rate * self.embedding_size, self.embedding_size]))
        self.assign(ospark.weight.truncated_normal(
                                obj_name="high_dimensional_bias",
                                weight_shape=[self.scale_rate * self.embedding_size]))
        self.assign(ospark.weight.truncated_normal(
                                obj_name="low_dimensional_bias",
                                weight_shape=[self.embedding_size]))

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        main_output = self.feedforward(input_data)
        residual_output = self.residual_net(input_data)
        added_residual = tf.add(main_output, residual_output)
        normalization_output = self.normalization(added_residual)
        return normalization_output

    def feedforward(self, input_data: tf.Tensor) -> tf.Tensor:
        mapping2high_dimensional = tf.matmul(input_data, self.assigned.mapping2high_dimensional) + self.assigned.high_dimensional_bias
        activated_outputs = self.activation(mapping2high_dimensional)
        mapping2low_dimensional = tf.matmul(activated_outputs, self.assigned.mapping2low_dimensional) + self.assigned.low_dimensional_bias
        return mapping2low_dimensional

    def residual_net(self, input_data: tf.Tensor) -> tf.Tensor:
        return input_data
