from ospark.nn.layers.normalization import Normalization
from ospark.nn.layers.activation import Activation
from typing import NoReturn, Optional
from . import Layer
import ospark.utility.weight_initializer
import tensorflow as tf
import ospark


class FeedForwardLayer(Layer):

    def __init__(self, 
                 obj_name: str, 
                 embedding_size: int, 
                 scale_rate: int,
                 dropout_rate: float,
                 mode: Optional[int]=None,
                 training_phase: Optional[bool]=None,
                 is_training: Optional[bool]=None,
                 activation: Optional[Activation]=None) -> NoReturn:
        """
        Args:
            obj_name: str
            embedding_size: int
            scale_rate: int
            dropout_rate: float
            mode: Optional[int]
                Has two modes "pre-ln" and "post-ln", default is "pre-ln"
            training_phase: Optional[bool]
            is_training: Optional[bool]
            activation: Optional[Activation]
        """

        super().__init__(obj_name=obj_name, is_training=is_training, training_phase=training_phase)
        self._activation     = activation or ospark.nn.layers.activation.ReLU()
        self._embedding_size = embedding_size
        self._scale_rate     = scale_rate
        self._dropout_rate   = dropout_rate
        self._dropout_layer  = tf.keras.layers.Dropout(rate=dropout_rate)
        self._mode           = mode if mode is not None else "pre-ln"

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
        if self._mode == "pre-ln":
            input_data = self._norm(input_data=input_data)
        main_output      = self.feedforward(input_data=input_data)
        residual_output  = self.residual_net(input_data=input_data)
        output           = tf.add(self.dropout_layer(main_output, training=self.training_phase), residual_output)
        if self._mode == "post-ln":
            output = self._norm(output)
        return output

    def feedforward(self, input_data: tf.Tensor) -> tf.Tensor:
        mapping2high_dimensional = tf.matmul(input_data, self._mapping2high_dimensional) + self._high_dimensional_bias
        activated_outputs        = self.activation(mapping2high_dimensional)
        mapping2low_dimensional  = tf.matmul(activated_outputs, self._mapping2low_dimensional) + self._low_dimensional_bias
        return mapping2low_dimensional

    def residual_net(self, input_data: tf.Tensor) -> tf.Tensor:
        return input_data
