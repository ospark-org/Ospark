from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import tensorflow as tf 
from enum import Enum

class LearningRateOption(Enum):

    transformer_warmup = lambda model_dimension, warmup_step:TransformerWarmup(model_dimension, warmup_step)


class TransformerWarmup(LearningRateSchedule):

    def __init__(self, model_dimension: int, warmup_step: float=4000.):
        self.model_dimension = tf.cast(model_dimension, dtype=tf.float32)
        self.warmup_step     = warmup_step

    def __call__(self, step):
        learning_rate = 1. / tf.sqrt(self.model_dimension) * \
                        tf.math.minimum(1 / tf.sqrt(step), step * tf.pow(self.warmup_step, -1.5))
        return learning_rate
