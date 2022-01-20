from ospark.nn.layers import Layer
from typing import List, NoReturn, Optional, Callable, Tuple, Union
from ospark.nn.component.activation import Activation, PassActivation
from functools import reduce
import tensorflow as tf
import ospark


class DenseLayer(Layer):

    def __init__(self,
                 obj_name:str,
                 input_dimension: int,
                 hidden_dimension: List[int],
                 activation: Optional[str]=None,
                 use_bias: Optional[bool]=True):
        super().__init__(obj_name=obj_name)
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
            weight = ospark.weight.glorot_uniform(obj_name=name, weight_shape=[input_dimension, output_dimension])
            self.assign(component=weight)
            if self.use_bias:
                bias = ospark.weight.zeros(obj_name=name + "_bias", weight_shape=[output_dimension])
                self.assign(component=bias)

    def bias_forward(self, input_data: tf.Tensor, weight: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        return self.activation(tf.matmul(input_data, weight[0]) + weight[1])

    def no_bias_forward(self, input_data: tf.Tensor, weight: tf.Tensor) -> tf.Tensor:
        return self.activation(tf.matmul(input_data, weight))

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        layers = [(self.assigned.__getattribute__(name=name), self.assigned.__getattribute__(name=name + "_bias"))
                  if self.use_bias else self.assigned.__getattribute__(name=name)
                  for name in self.layers_name]
        output = reduce(self.forward, layers, input_data)
        return output