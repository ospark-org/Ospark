import tensorflow as tf
from typing import NoReturn

class LossFunction:

    def __init_subclass__(cls) -> NoReturn:
        super().__init_subclass__()
        setattr(LossFunction, cls.__name__, cls)

    def calculate(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def __call__(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        return self.calculate(prediction, target_data)


class CrossEntropy(LossFunction):

    def calculate(self, prediction, target_data):
        return -tf.reduce_sum(target_data * tf.math.log(prediction))