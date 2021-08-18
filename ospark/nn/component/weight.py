from __future__ import annotations
import tensorflow as tf 
from typing import List, NoReturn


class Weight:

    def __init__(self, 
                 obj_name: str, 
                 creator_func,
                 weight_shape: list) -> NoReturn:
        self._obj_name         = obj_name
        self._indexed_name = None
        self._value        = None
        self._creator_func = creator_func
        self.weight_shape  = weight_shape

    @property
    def obj_name(self) -> str:
        return self._obj_name
    
    @property
    def indexed_name(self) -> NoReturn:
        return self._indexed_name

    @property
    def value(self) -> NoReturn:
        return self._value

    @property
    def creator_func(self):
        return self._creator_func

    def create(self, prefix_word: str) -> NoReturn:
        manager = WeightOperator()
        self._indexed_name = prefix_word + "_" + self.obj_name
        manager.add_weight(self)
        if self._value is None:
            print(f"Initialize weight {self.indexed_name}")
            self._value = tf.Variable(self.creator_func(shape=self.weight_shape))

    @property
    def get(self) -> NoReturn:
        return (self.indexed_name, self.value.numpy().tolist())

    def restore(self, weight: tf.Tensor) -> NoReturn:
        if weight is not None:
            if all(tf.shape(weight) == tf.constant(self.weight_shape)):
                self._value = tf.Variable(weight)
            else:
                print(f"Weight \"{self.indexed_name}\", shape does not match the original setting, so use the initialization weight")


class WeightOperator:
    _instance       = None
    _fitst_initial  = False
    _restore_weight = {}

    def __new__(cls) -> WeightOperator:
        if  not cls._instance:
            WeightOperator._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> NoReturn:
        if not self._fitst_initial:
            self.weights = []
            WeightOperator._fitst_initial = True
    
    def add_weight(self, weight: Weight) -> NoReturn:
        self.weights.append(weight)
        if weight.indexed_name in self._restore_weight:
            weight.restore(self._restore_weight.pop(weight.indexed_name))

    @classmethod
    def restore(cls, weight: dict) -> NoReturn:
        cls._restore_weight = weight
        
    @property
    def collect(self) -> List[tf.Tensor]:
        return [weight.value for weight in self.weights]

    @property
    def get(self) -> dict:
        return dict([weight.get for weight in self.weights])

    @classmethod
    def release_loaded_weight(cls) -> NoReturn:
        del cls._restore_weight


def truncated_normal(obj_name, weight_shape) -> Weight:
    return Weight(obj_name, tf.random.truncated_normal, weight_shape)

def normal(obj_name, weight_shape) -> Weight:
    return Weight(obj_name, tf.random.normal, weight_shape)

def uniform(obj_name, weight_shape) -> Weight:
    return Weight(obj_name, tf.random.uniform, weight_shape)