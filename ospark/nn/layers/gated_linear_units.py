from ospark.nn.layers.activation import PassActivation, ReLU, Swish, Sigmoid, Activation
from ospark import weight_initializer
from typing import Optional, NoReturn
import tensorflow as tf


class Bilinear(Activation):

    def __init__(self,
                 dimension: int,
                 is_training: Optional[bool]):
        super(Bilinear, self).__init__(is_training=is_training)
        self._dimension = dimension

        self._weight_1 = weight_initializer.glorot_uniform(obj_name="weight_1", shape=[dimension, dimension])
        self._weight_2 = weight_initializer.glorot_uniform(obj_name="weight_2", shape=[dimension, dimension])

        self._bias_1 = weight_initializer.zeros(obj_name="bias_1", shape=[dimension])
        self._bias_2 = weight_initializer.zeros(obj_name="bias_2", shape=[dimension])

        self.activation_function = PassActivation()

    @property
    def activation_function(self) -> Activation:
        return self._activation_function

    @activation_function.setter
    def activation_function(self, value: Activation) -> NoReturn:
        self._activation_function = value

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        linear_layer_1 = self.activation_function(tf.matmul(input_data, self._weight_1) + self._bias_1)
        linear_layer_2 = tf.matmul(input_data, self._weight_2) + self._bias_2
        return tf.multiply(linear_layer_1, linear_layer_2)


class GLU(Bilinear):

    def __init__(self,
                 dimension: int,
                 is_training: Optional[bool]):
        super(GLU, self).__init__(dimension=dimension, is_training=is_training)

        self.activation_function = Sigmoid()

class ReGLU(GLU):

    def __init__(self,
                 dimension: int,
                 is_training: Optional[bool]):
        super(ReGLU, self).__init__(dimension=dimension, is_training=is_training)

        self.activation_function = ReLU()

class SwiGLU(GLU):

    def __init__(self,
                 dimension: int,
                 beta: Optional[float]=None,
                 is_training: Optional[bool]=None):
        super(SwiGLU, self).__init__(dimension=dimension, is_training=is_training)
        self._beta               = beta
        self.activation_function = Swish(beta=beta or 1.)


