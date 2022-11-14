import tensorflow as tf

from ospark.nn.loss_function.loss_function import LossFunction


class Dice(LossFunction):

    def calculate(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        eps = 1e-5
        intersection = tf.reduce_sum(tf.multiply(prediction, target_data))
        union = tf.reduce_sum(prediction) + tf.reduce_sum(target_data) + eps
        loss  = 1. - 2 * intersection / union
        return loss