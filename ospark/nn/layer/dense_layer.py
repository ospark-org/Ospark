from ospark.nn.layer import Layer
from typing import List, NoReturn
from functools import reduce
import tensorflow as tf
import ospark

class DenseLayer(Layer):

    def __init__(self,
                 obj_name:str,
                 input_dimension: int,
                 hidden_dimension: List[int]):
        super().__init__(obj_name=obj_name)
        self._input_dimension  = input_dimension
        self._hidden_dimension = hidden_dimension
        self._layers_name      = []

    @property
    def input_dimension(self) -> int:
        return self._input_dimension

    @property
    def hidden_dimension(self) -> List[int]:
        return self._hidden_dimension

    @property
    def layers_name(self) -> list:
        return self._layers_name

    def on_creating(self) -> NoReturn:
        for i, output_dimension in enumerate(self.hidden_dimension):
            if i == 0:
                input_dimension = self.input_dimension
            else:
                input_dimension = self.hidden_dimension[i - 1]
            name = f"layer_{i}"
            self._layers_name.append(name)
            weight = ospark.weight.glorot_uniform(obj_name=name, weight_shape=[input_dimension, output_dimension])
            bias   = ospark.weight.zeros(obj_name=name+"_bias", weight_shape=[output_dimension])
            self.assign(component=weight)
            self.assign(component=bias)

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        layers = [(self.assigned.__getattribute__(name=name), self.assigned.__getattribute__(name=name + "_bias"))
                  for name in self.layers_name]
        output = reduce(lambda output, weight: tf.matmul(output, weight[0]) + weight[1], layers, input_data)
        return output
