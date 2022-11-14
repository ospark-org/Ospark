from __future__ import annotations
import tensorflow as tf
from ospark.nn.layers import Layer
from typing import Optional
from typing import NoReturn


class Activation(Layer):

    def __init__(self, is_training: Optional[bool]=None):
        super(Activation, self).__init__(obj_name=self.__class__.__name__, is_training=is_training)

    def __init_subclass__(cls) -> NoReturn:
        super().__init_subclass__()
        setattr(Activation, cls.__name__, cls)
    
    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        return self.pipeline(input_data=input_data)


class PassActivation(Activation):

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        return input_data


class ReLU(Activation):

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        return tf.nn.relu(input_data)


class ELU(Activation):

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        return tf.nn.elu(input_data)


class LeakyReLU(Activation):

    @property
    def alpha(self) -> tf.float32:
        return self.__alpha

    def __init__(self, alpha: tf.float32) -> NoReturn:
        self.__alpha = alpha
        super().__init__()

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        return tf.nn.leaky_relu(input_data, self.alpha)


class SELU(Activation):

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        return tf.nn.selu(input_data)


class CReLU(Activation):

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        return tf.nn.crelu(input_data)


class GELU(Activation):

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        return tf.nn.gelu(input_data)


class Swish(Activation):

    def __init__(self, beta: float):
        super(Swish, self).__init__()
        self._beta = tf.constant(beta)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        return tf.nn.swish(input_data, beta=self._beta)


class Sigmoid(Activation):

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        return tf.nn.sigmoid(input_data)