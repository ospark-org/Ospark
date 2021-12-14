from . import Layer
from typing import NoReturn, Optional
from ospark.nn.component.activation import Activation
from ospark.nn.component.normalization import Normalization, BatchNormalization
import tensorflow as tf
import ospark


class DistillingLayer(Layer):

    def __init__(self, 
                 obj_name: str,
                 embedding_size: int,
                 filter_width: Optional[int]=None,
                 pooling=None,
                 pooling_size: Optional[list]=None,
                 strides: Optional[list]=None,
                 activation: Optional[Activation]=None,
                 norm: Optional[Normalization]=None):
        super().__init__(obj_name=obj_name)
        self._embedding_size = embedding_size
        self._filter_width   = filter_width or 3
        self._pooling_size   = pooling_size or [1, 3, 1]
        self._strides        = strides or [1, 2, 1]
        self._norm           = norm or BatchNormalization(input_depth=embedding_size)
        self._activation     = activation or ospark.activation.elu()
        self._pooling        = pooling or tf.nn.max_pool1d

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def activation(self) -> Activation:
        return self._activation

    @property
    def pooling(self):
        return self._pooling

    @property
    def strides(self) -> list:
        return self._strides

    @property
    def filter_width(self) ->int:
        return self._filter_width

    @property
    def pooling_size(self) -> list:
        return self._pooling_size

    @property
    def norm(self) -> Normalization:
        return self._norm

    def on_creating(self) -> NoReturn:
        self.assign(component=ospark.weight.truncated_normal(obj_name="conv_filter",
                                                             weight_shape=[self.filter_width, 
                                                                           self.embedding_size, 
                                                                           self.embedding_size]))
        self.assign(component=self.norm, name="norm")

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        conv_output = tf.nn.conv1d(input_data, self.assigned.conv_filter, [1, 1, 1], padding="SAME")
        norm_output = self.assigned.norm(conv_output)
        activation  = self.activation(norm_output)
        pooling     = self.pooling(activation, self.pooling_size, self.strides, padding="SAME")
        return pooling