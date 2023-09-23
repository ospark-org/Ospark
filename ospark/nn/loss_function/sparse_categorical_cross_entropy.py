import tensorflow as tf
from typing import Optional
from ospark.nn.loss_function.loss_function import LossFunction


class SparseCategoricalCrossEntropy(LossFunction):

    def __init__(self, from_logits: Optional[bool]=None, ignore_class: Optional[int]=None):
        from_logits = False if from_logits is None else True
        self._loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits,
                                                                            ignore_class=ignore_class,
                                                                            reduction='none')

    @property
    def loss_function(self) -> tf.keras.losses.Loss:
        return self._loss_function

    def calculate(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        mask  = tf.logical_not(tf.math.equal(target_data, 0))
        loss  = self.loss_function(target_data, prediction)
        mask  = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)