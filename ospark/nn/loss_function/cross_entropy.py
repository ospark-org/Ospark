import tensorflow as tf
from typing import Optional
from ospark.nn.loss_function.loss_function import LossFunction


class CrossEntropy(LossFunction):

    def __init__(self, reduction: Optional[str]=None):
        """
        Args
            reduction: Optional[str]
                There are two options are mean and sum, the default is mean.
        """

        self._reduction = reduction or "mean"
        if self._reduction != "mean" and self._reduction != "sum":
            raise AttributeError(f"reduction must be mean or sum, doesn't have {reduction}")

    def calculate(self,
                  prediction: tf.Tensor,
                  target_data: tf.Tensor,
                  mask_matrix: Optional[tf.Tensor]=None) -> tf.Tensor:
        loss_value = target_data * tf.math.log(prediction)
        if mask_matrix is not None:
            loss_value *= mask_matrix
        if self._reduction == "sum":
            return -tf.reduce_sum(loss_value)
        else:
            if mask_matrix is not None:
                elements_number = tf.reduce_sum(mask_matrix)
                return -tf.reduce_sum(loss_value) / elements_number
            else:
                return -tf.reduce_mean(loss_value)