import tensorflow as tf

from ospark.nn.loss_function.loss_function import LossFunction


class Balanced(LossFunction):

    def __init__(self, image_width: int, image_height: int, omega: int):
        self._image_width  = image_width
        self._image_height = image_height
        self._omega        = omega

    @property
    def image_width(self) -> int:
        return self._image_width

    @property
    def image_height(self) -> int:
        return self._image_height

    @property
    def omega(self) -> int:
        return self._omega

    def calculate_loss(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        total_elements = self.image_width * self.image_height
        beta = 1 - (self.omega / total_elements)
        reward_part = tf.multiply(beta, tf.multiply(target_data, tf.math.log(prediction)))
        punish_part = tf.multiply((1. - beta), tf.multiply((1. - target_data), tf.math.log((1. - prediction))))
        loss = -tf.reduce_mean(tf.add(reward_part, punish_part))
        return loss