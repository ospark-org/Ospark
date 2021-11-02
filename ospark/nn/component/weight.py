from __future__ import annotations
import tensorflow as tf 
from typing import List, NoReturn, Optional, Tuple


class Weight:

    def __init__(self, 
                 obj_name: str,
                 initial_value: tf.Tensor,
                 trainable: Optional[bool]=True) -> NoReturn:
        self._obj_name      = obj_name
        self._indexed_name  = None
        self._value         = None
        self._initial_value = initial_value
        self._trainable     = trainable

    @property
    def obj_name(self) -> str:
        return self._obj_name
    
    @property
    def indexed_name(self) -> None:
        return self._indexed_name

    @property
    def value(self) -> None:
        return self._value

    @property
    def initial_value(self) -> tf.Tensor:
        return self._initial_value

    @property
    def trainable(self) -> bool:
        return self._trainable

    def create(self, prefix_word: str) -> NoReturn:
        manager = WeightOperator()
        self._indexed_name = prefix_word + "_" + self.obj_name
        manager.add_weight(self)
        if self._value is None:
            print(f"Initialize weight {self.indexed_name}")
            self._value = tf.Variable(self.initial_value, trainable=self.trainable)

    @property
    def get(self) -> Tuple[str, list]:
        return self.indexed_name, self.value.numpy().tolist()

    def restore(self, weight: tf.Tensor) -> NoReturn:
        if weight is not None:
            if all(tf.shape(weight) == tf.shape(self.initial_value)):
                self._value = tf.Variable(weight)
            else:
                print(f"Weight \"{self.indexed_name}\", shape does not match the original setting, so use the initialization weight")


class WeightOperator:
    _instance        = None
    _fitst_initial   = False
    _restore_weights = {}

    def __new__(cls) -> WeightOperator:
        if not cls._instance:
            WeightOperator._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> NoReturn:
        if not self._fitst_initial:
            self._weights = []
            WeightOperator._fitst_initial = True
    
    def add_weight(self, weight: Weight) -> NoReturn:
        self._weights.append(weight)
        if weight.indexed_name in self._restore_weights:
            weight.restore(self._restore_weights.pop(weight.indexed_name))

    @classmethod
    def restore(cls, weights: dict) -> NoReturn:
        cls._restore_weights = weights

    def collect(self, partition_name: Optional[str]=None) -> List[tf.Tensor]:
        if partition_name is None:
            return [weight.value for weight in self._weights if weight.trainable]
        else:
            partition_weight = [weight.value for weight in self._weights if partition_name in weight.indexed_name and weight.trainable]
            if partition_weight == []:
                raise NameError(f"partition name {partition_name} is not exist, please check")
            else:
                return partition_weight

    @property
    def get(self) -> dict:
        return dict([weight.get for weight in self._weights])

    @classmethod
    def release_loaded_weight(cls) -> NoReturn:
        del cls._restore_weights


def truncated_normal(obj_name, weight_shape) -> Weight:
    return Weight(obj_name, tf.random.truncated_normal(weight_shape))


def normal(obj_name, weight_shape) -> Weight:
    return Weight(obj_name, tf.random.normal(weight_shape))


def uniform(obj_name, weight_shape) -> Weight:
    return Weight(obj_name, tf.random.uniform(weight_shape))


def ones(obj_name, weight_shape) -> Weight:
    return Weight(obj_name, tf.ones(weight_shape))


def zeros(obj_name, weight_shape) -> Weight:
    return Weight(obj_name, tf.zeros(weight_shape))

def glorot_uniform(obj_name, weight_shape) -> Weight:
    limit = tf.cast(tf.sqrt(6 / (weight_shape[-1] + weight_shape[-2])), dtype=tf.float32)
    return Weight(obj_name, tf.random.uniform(weight_shape, minval=-limit, maxval=limit))