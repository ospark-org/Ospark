from .loss_function import LossFunction
from typing import Optional
import tensorflow as tf


class ContrastiveLoss(LossFunction):

    def __init__(self, temperature: Optional[float]=None):
        self._temperature = temperature or 1.

    def calculate(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        matmul_output = tf.matmul(prediction, target_data, transpose_b=True)

        part_1_loss = tf.math.log(tf.exp(tf.linalg.diag_part(matmul_output) / self._temperature) / tf.reduce_sum(tf.exp(matmul_output / self._temperature), axis=1))
        part_2_loss =  tf.math.log(tf.exp(tf.linalg.diag_part(matmul_output) / self._temperature) / tf.reduce_sum(tf.exp(tf.transpose(matmul_output) / self._temperature), axis=1))
        loss = -1 * tf.reduce_mean(part_1_loss + part_2_loss)
        return loss