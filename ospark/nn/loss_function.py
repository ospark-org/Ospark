import tensorflow as tf
from typing import NoReturn, Optional, Tuple

class LossFunction:

    def __init_subclass__(cls) -> NoReturn:
        super().__init_subclass__()
        setattr(LossFunction, cls.__name__, cls)

    def calculate(self, *args, **kwargs) -> tf.Tensor:
        raise NotImplementedError()

    def __call__(self, prediction: tf.Tensor, target_data: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return self.calculate(prediction, target_data)


class SparseCategoricalCrossEntropy(LossFunction):

    def __init__(self):
        self._loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    @property
    def loss_function(self) -> tf.keras.losses.Loss:
        return self._loss_function

    def calculate(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        mask  = tf.logical_not(tf.math.equal(target_data, 0))
        loss  = self.loss_function(target_data, prediction)
        mask  = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


class CrossEntropy(LossFunction):

    def calculate(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        return -tf.reduce_sum(target_data * tf.math.log(prediction))


class Dice(LossFunction):

    def calculate(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        eps = 1e-5
        intersection = tf.reduce_sum(tf.multiply(prediction, target_data))
        union = tf.reduce_sum(prediction) + tf.reduce_sum(target_data) + eps
        loss  = 1. - 2 * intersection / union
        return loss


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


class IoU(LossFunction):

    def calculate(self,
                  prediction: tf.Tensor,
                  target_data: tf.Tensor,
                  cls_target: tf.Tensor) -> tf.Tensor:
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
                 target_data: tf.Tensor,
                 cls_target: tf.Tensor) -> tf.Tensor:
        loss_value = self.calculate(prediction=prediction, target_data=target_data, cls_target=cls_target)
        return loss_value


class Degree(LossFunction):

    def __init__(self, coefficient: Optional[float]=1.0):
        self._coefficient = coefficient

    @property
    def coefficient(self) -> float:
        return self._coefficient

    def calculate(self,
                  prediction: tf.Tensor,
                  target_data: tf.Tensor,
                  cls_target: tf.Tensor) -> tf.Tensor:
        loss = (1.0 - tf.math.cos(prediction * cls_target - target_data))

        degree_loss = self.coefficient * tf.divide(tf.reduce_sum(loss), (tf.reduce_sum(cls_target) + 1))
        return degree_loss

    def __call__(self,
                 prediction: tf.Tensor,
                 target_data: tf.Tensor,
                 cls_target: tf.Tensor) -> tf.Tensor:
        loss_value = self.calculate(prediction=prediction, target_data=target_data, cls_target=cls_target)
        return loss_value