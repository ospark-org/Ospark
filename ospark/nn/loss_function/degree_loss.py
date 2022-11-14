from typing import Optional

import tensorflow as tf

from ospark.nn.loss_function.loss_function import LossFunction


class Degree(LossFunction):

    def __init__(self, coefficient: Optional[float]=1.0):
        self._coefficient = coefficient

    @property
    def coefficient(self) -> float:
        return self._coefficient

    def calculate(self,
                  prediction: tf.Tensor,
                  target_data: tf.Tensor) -> tf.Tensor:
        cls_target = tf.cast(tf.where(target_data != 0), tf.float32)
        loss = (1.0 - tf.math.cos(prediction * cls_target - target_data))
        degree_loss = self.coefficient * tf.divide(tf.reduce_sum(loss), (tf.reduce_sum(cls_target) + 1))
        return degree_loss

    def __call__(self,
                 prediction: tf.Tensor,
                 target_data: tf.Tensor) -> tf.Tensor:
        loss_value = self.calculate(prediction=prediction, target_data=target_data)
        return loss_value