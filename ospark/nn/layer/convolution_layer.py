from __future__ import annotations
from ospark.nn.layer import Layer
from ospark.nn.component.activation import Activation, relu, PassActivation
from ospark.nn.component.normalization import Normalization, BatchNormalization, PassNormalization
from typing import List, NoReturn, Optional
import tensorflow as tf
import ospark


class ConvolutionLayer(Layer):

    def __init__(self,
                 obj_name: str,
                 filter_size: List[int],
                 strides: List[int],
                 padding: str,
                 activation: Optional[Activation]=None,
                 normalization: Optional[Normalization]=None,
                 layer_order: Optional[List[str]]=None) -> NoReturn:
        super().__init__(obj_name=obj_name)
        self._filter_size   = filter_size
        self._strides       = strides
        self._padding       = padding
        self._activation    = activation or PassActivation()
        self._normalization = normalization or PassNormalization()
        self._layer_order   = layer_order or ["conv", "norm", "activate"]

    @property
    def filter_size(self) -> List[int]:
        return self._filter_size

    @property
    def strides(self) -> List[int]:
        return self._strides

    @property
    def padding(self) -> str:
        return self._padding

    @property
    def activation(self) -> Activation:
        return self._activation

    @property
    def normalization(self) -> Normalization:
        return self._normalization

    @property
    def layer_order(self) -> List[str]:
        return self._layer_order

    def set_norm(self, norm: Normalization) -> NoReturn:
        self._normalization = norm

    def set_activate(self, activation: Activation) -> NoReturn:
        self._activation = activation

    def set_order(self, layer_order: List[str]) -> NoReturn:
        self._layer_order = layer_order

    @classmethod
    def bn_relu_conv(cls,
                     obj_name: str,
                     filter_size: List[int],
                     strides: List[int],
                     padding: str,
                     trainable: bool) -> ConvolutionLayer:
        activation    = relu()
        normalization = BatchNormalization(input_depth=filter_size[-2], trainable=trainable)
        return cls(obj_name=obj_name,
                   filter_size=filter_size,
                   strides=strides,
                   padding=padding,
                   activation=activation,
                   normalization=normalization,
                   layer_order=["norm", "activate", "conv"])

    @classmethod
    def conv_bn_relu(cls,
                     obj_name: str,
                     filter_size: List[int],
                     strides: List[int],
                     padding: str,
                     trainable: bool) -> ConvolutionLayer:
        activation = relu()
        normalization = BatchNormalization(input_depth=filter_size[-1], trainable=trainable)
        return cls(obj_name=obj_name,
                   filter_size=filter_size,
                   strides=strides,
                   padding=padding,
                   activation=activation,
                   normalization=normalization,
                   layer_order=["conv", "norm", "activate"])

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        output = input_data
        for layer_name in self.layer_order:
            output = self.__getattribute__(layer_name)(output)
        return output

    def on_creating(self) -> NoReturn:
        self.assign(component=ospark.weight.glorot_uniform(obj_name="filter", weight_shape=self.filter_size), name="filter")
        self.assign(component=self.normalization, name="norm")

    def conv(self, input_data: tf.Tensor) -> tf.Tensor:
        output = tf.nn.conv2d(input=input_data, filters=self.assigned.filter, strides=self.strides, padding=self.padding)
        return output

    def norm(self, input_data: tf.Tensor) -> tf.Tensor:
        output = self.assigned.norm(input_data)
        return output

    def activate(self, input_data: tf.Tensor) -> tf.Tensor:
        output = self.activation(input_data)
        return output
