import tensorflow as tf
from typing import NoReturn

class Normalization:

    def __init_subclass__(cls) -> NoReturn:
        super().__init_subclass__()
        setattr(Normalization, cls.__name__, cls)

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        return self.calculate(input_data)


class PassNormalization(Normalization):

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        return input_data


class LayerNormalization(Normalization):

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        mean, variance = tf.nn.moments(input_data, axes=[1], keepdims=True)
        normalization_outputs = (input_data - mean) / tf.sqrt(variance + 1e-5)
        return normalization_outputs