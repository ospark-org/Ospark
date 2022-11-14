import tensorflow as tf

from ospark.nn.loss_function.loss_function import LossFunction


class Focal(LossFunction):

    def __init__(self, alpha: float, beta: float):
        self._alpha  = alpha
        self._beta   = beta

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    def calculate_loss(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        A_part = self.alpha * target_data * tf.math.pow(1 - prediction, self.gamma) * tf.math.log(prediction)
        B_part = (1 - self.alpha) * (1 - target_data) * tf.math.pow(prediction, self.gamma) * tf.math.log(1 - prediction)
        loss = -tf.reduce_mean(A_part + B_part)
        return loss