from typing import List, Optional
import tensorflow as tf
from ospark import Weight


class Glorot(Weight):

    def init_weight(self) -> tf.Tensor:
        limit  = tf.cast(tf.sqrt(6 / (self.shape[-1] + self.shape[-2])), dtype=tf.float32)
        weight = tf.random.uniform(self.shape, minval=-limit, maxval=limit)
        return weight


class Zeros(Weight):

    def init_weight(self) -> tf.Tensor:
        weight = tf.zeros(self.shape)
        return weight


class Ones(Weight):

    def init_weight(self) -> tf.Tensor:
        weight = tf.ones(shape=self.shape)
        return weight


class TruncatedNormal(Weight):

    def init_weight(self) -> tf.Tensor:
        weight = tf.random.truncated_normal(shape=self.shape)
        return weight


class Normal(Weight):

    def init_weight(self) -> tf.Tensor:
        weight = tf.random.normal(shape=self.shape)
        return weight


class Uniform(Weight):

    def init_weight(self) -> tf.Tensor:
        weight = tf.random.uniform(shape=self.shape)
        return weight


def truncated_normal(obj_name: str, shape: List[int], trainable: Optional[bool]=None) -> Weight:
    return TruncatedNormal(obj_name=obj_name, shape=shape, trainable=trainable)


def normal(obj_name: str, shape: List[int], trainable: Optional[bool]=None) -> Weight:
    return Normal(obj_name=obj_name, shape=shape, trainable=trainable)


def uniform(obj_name: str, shape: List[int], trainable: Optional[bool]=None) -> Weight:
    return Uniform(obj_name=obj_name, shape=shape, trainable=trainable)


def ones(obj_name: str, shape: List[int], trainable: Optional[bool]=None) -> Weight:
    return Ones(obj_name=obj_name, shape=shape, trainable=trainable)


def zeros(obj_name: str, shape: List[int], trainable: Optional[bool]=None) -> Weight:
    return Zeros(obj_name=obj_name, shape=shape, trainable=trainable)


def glorot_uniform(obj_name: str, shape: List[int], trainable: Optional[bool]=None) -> Weight:
    return Glorot(obj_name=obj_name, shape=shape, trainable=trainable)