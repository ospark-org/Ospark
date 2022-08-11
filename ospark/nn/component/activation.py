import tensorflow as tf 
from typing import NoReturn

class Activation:

    def __init_subclass__(cls) -> NoReturn:
        super().__init_subclass__()
        setattr(Activation, cls.__name__, cls)
    
    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        return self.calculate(input_data=input_data)


class PassActivation(Activation):

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        return input_data


class ReLU(Activation):

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        return tf.nn.relu(input_data)


class ELU(Activation):

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        return tf.nn.elu(input_data)


class LeakyReLU(Activation):

    @property
    def alpha(self) -> tf.float32:
        return self.__alpha

    def __init__(self, alpha: tf.float32) -> NoReturn:
        self.__alpha = alpha
        super().__init__()

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        return tf.nn.leaky_relu(input_data, self.alpha)


class SELU(Activation):

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        return tf.nn.selu(input_data)


class CReLU(Activation):

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        return tf.nn.crelu(input_data)


class GELU(Activation):

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        return tf.nn.gelu(input_data)