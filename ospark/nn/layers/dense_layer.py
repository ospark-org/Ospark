import ospark.utility.weight_initializer
from ospark.nn.layers import Layer
from typing import List, NoReturn, Optional, Callable, Tuple, Union
from ospark.nn.component.activation import Activation, PassActivation
from functools import reduce
from ospark import Weight
import tensorflow as tf
import ospark


class DenseLayer(Layer):

    def __init__(self,
                 obj_name:str,
                 input_dimension: int,
                 hidden_dimension: List[int],
                 is_training: Optional[bool]=None,
                 activation: Optional[str]=None,
                 use_bias: Optional[bool]=True):
        super().__init__(obj_name=obj_name, is_training=is_training)
        self._input_dimension  = input_dimension
        self._hidden_dimension = hidden_dimension
        self._layers_name      = []
        self._activation       = PassActivation() if activation is None else getattr(Activation, activation.lower())()
        self._use_bias         = use_bias
        self._forward          = self.bias_forward if use_bias else self.no_bias_forward

    @property
    def input_dimension(self) -> int:
        return self._input_dimension

    @property
    def hidden_dimension(self) -> List[int]:
        return self._hidden_dimension

    @property
    def layers_name(self) -> list:
        return self._layers_name

    @property
    def activation(self) -> Callable[[tf.Tensor], tf.Tensor]:
        return self._activation

    @property
    def use_bias(self) -> bool:
        return self._use_bias

    @property
    def forward(self) -> Callable[[tf.Tensor, Union[List[tf.Tensor], tf.Tensor]], tf.Tensor]:
        return self._forward

    def in_creating(self) -> NoReturn:
        for i, output_dimension in enumerate(self.hidden_dimension):
            if i == 0:
                input_dimension = self.input_dimension
            else:
                input_dimension = self.hidden_dimension[i - 1]
            name = f"layer_{i}"
            self._layers_name.append(name)
            setattr(self, name, ospark.utility.weight_initializer.glorot_uniform(obj_name=name, shape=[input_dimension, output_dimension]))
            if self.use_bias:
                setattr(self, name + "_bias", ospark.utility.weight_initializer.zeros(obj_name=name + "_bias", shape=[output_dimension]))

    def bias_forward(self, input_data: tf.Tensor, weight: Tuple[Weight, Weight]) -> tf.Tensor:
        return self.activation(tf.matmul(input_data, weight[0]) + weight[1])

    def no_bias_forward(self, input_data: tf.Tensor, weight: tf.Tensor) -> tf.Tensor:
        return self.activation(tf.matmul(input_data, weight))

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        layers = [(getattr(self, name), getattr(self, name + "_bias"))
                  if self.use_bias else getattr(self, name)
                  for name in self.layers_name]
        output = reduce(self.forward, layers, input_data)
        return output