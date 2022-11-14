from typing import Tuple

import tensorflow as tf

from ospark.nn.loss_function.loss_function import LossFunction


class IoU(LossFunction):

    def calculate(self,
                  prediction: tf.Tensor,
                  target_data: tf.Tensor) -> tf.Tensor:
        cls_target = tf.cast(tf.where(target_data != 0), tf.float32)
        iou  = self.calculate_iou(prediction, target_data, cls_target)
        loss = -tf.math.log((tf.reduce_sum(iou, axis=[1, 2, 3]) + 1) / (tf.reduce_sum(cls_target, axis=[1, 2, 3]) + 1))
        return tf.reduce_mean(loss)

    def calculate_iou(self,
                      prediction: tf.Tensor,
                      target_data: tf.Tensor,
                      cls_target: tf.Tensor) -> tf.Tensor:
        top_prediction, right_prediction, bottom_prediction, left_prediction = self._slice_channel(prediction)

        top_target, right_target, bottom_target, left_target = self._slice_channel(target_data)

        training_area  = tf.multiply(tf.add(top_prediction, bottom_prediction),
                                     tf.add(right_prediction, left_prediction))
        target_area    = tf.multiply(tf.add(top_target, bottom_target), tf.add(right_target, left_target))
        intersection_h = tf.add(tf.minimum(top_prediction, top_target), tf.minimum(bottom_prediction, bottom_target))
        intersection_w = tf.add(tf.minimum(right_prediction, right_target), tf.minimum(left_prediction, left_target))
        intersection   = tf.multiply(intersection_h, intersection_w)

        union = tf.subtract(tf.add(training_area, target_area), intersection)
        iou   = tf.maximum(tf.divide(intersection + 1, tf.add(union, 1)), 0)
        return iou * cls_target

    def _slice_channel(self, input_data: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        top    = input_data[:, :, :, 0:1]
        right  = input_data[:, :, :, 1:2]
        bottom = input_data[:, :, :, 2:3]
        left   = input_data[:, :, :, 3:4]
        return top, right, bottom, left

    def __call__(self,
                 prediction: tf.Tensor,
                 target_data: tf.Tensor) -> tf.Tensor:
        loss_value = self.calculate(prediction=prediction, target_data=target_data)
        return loss_value