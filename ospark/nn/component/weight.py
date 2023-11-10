from __future__ import annotations
import tensorflow as tf 
from typing import List, NoReturn, Optional
from .name_space import NameSpace
import logging


class Weight:

    def __init__(self, 
                 obj_name: str,
                 shape: List[int],
                 trainable: Optional[bool]=None):
        self._obj_name           = obj_name
        self._indexed_name       = None
        self._value: tf.Variable = None
        self._shape              = shape
        self._scale              = tf.constant(1.0)
        self._name_space         = NameSpace("")
        if trainable is None:
            self._trainable          = True
            self._is_default_setting = True
        else:
            self._trainable          = trainable
            self._is_default_setting = False

    @property
    def obj_name(self) -> str:
        return self._obj_name
    
    @property
    def indexed_name(self) -> None:
        return self._indexed_name

    @property
    def value(self) -> tf.Variable:
        return self._value

    @property
    def shape(self) -> List[int]:
        return self._shape

    @property
    def trainable(self) -> bool:
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool):
        if self._value is not None:
            print(f"weight: {self.obj_name} is not change trainable attribute.")
        self._trainable = value

    def __mul__(self, other):
        if self._value is not None:
            self._value *= other
        else:
            self._scale = tf.constant(other)
        return self

    def __str__(self) -> str:
        if self._value is not None:
            return str(self._value.numpy().tolist())
        else:
            return str("value is None.")

    def init_weight(self) -> tf.Tensor:
        raise NotImplementedError("instance class Weight must implement init_weight.")

    def create(self) -> NoReturn:
        manager = WeightOperator()
        self._indexed_name = self._name_space._prefix.name + self.obj_name
        manager.add_weight(self)
        if self._value is None:
            #TODO debug mode.
            if manager._show_info:
                print(f"Initialize weight {self.indexed_name}.")
            self._value = tf.Variable(self.init_weight() * self._scale, trainable=self.trainable, name=self._indexed_name)

    def restore(self, weight: tf.Tensor) -> NoReturn:
        if weight is not None:
            if weight.shape == self.shape:
                self._value = tf.Variable(weight, trainable=self.trainable, name=self._indexed_name)
            else:
                print(f"Weight \"{self.indexed_name}\", shape does not match the original setting, so use the initialization weight")


class WeightOperator:
    _instance        = None
    _first_initial   = False
    _restore_weights = {}
    _show_info       = False

    def __new__(cls) -> WeightOperator:
        if not cls._instance:
            WeightOperator._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> NoReturn:
        if not self._first_initial:
            self._weights = []
            WeightOperator._first_initial = True

    def add_weight(self, weight: Weight) -> NoReturn:
        self._weights.append(weight)
        if weight.indexed_name in self._restore_weights:
            if self._show_info:
                print(f"Restore weight: {weight.indexed_name}")
            weight.restore(tf.convert_to_tensor(self._restore_weights.pop(weight.indexed_name), dtype=tf.float32))

    @classmethod
    def restore(cls, weights: dict) -> NoReturn:
        cls._restore_weights.update(weights)

    @classmethod
    def clear_restore_weights(cls) -> NoReturn:
        if cls._restore_weights != {}:
            logging.info(f"There are still {cls._restore_weights.keys()} in the temporary weight that has not been read and will be deleted soon.")
        cls._restore_weights.clear()

    def clear_weights(self):
        del self._weights
        self._weights = []

    def collect_weights(self, partition_name: Optional[str]=None) -> List[tf.Tensor]:
        if partition_name is None:
            return [weight.value for weight in self._weights if weight.trainable]
        else:
            partition_weight = [weight.value for weight in self._weights if partition_name in weight.indexed_name and weight.trainable]
            if partition_weight == []:
                raise NameError(f"partition name {partition_name} is not exist, please check")
            else:
                return partition_weight

    def collect_weights_name(self, partition_name: Optional[str]=None) -> List[str]:
        if partition_name is None:
            return [weight.indexed_name for weight in self._weights if weight.trainable]
        else:
            partition_weights_name = [weight.indexed_name for weight in self._weights if partition_name in weight.indexed_name and weight.trainable]
            if partition_weights_name == []:
                raise NameError(f"partition name {partition_name} is not exist, please check")
            else:
                return partition_weights_name

    def get_weights(self, partition_name: Optional[str]=None) -> dict:
        if partition_name is None:
            return self.weights
        else:
            partition_weights_name = {weight.indexed_name: weight.value.numpy().tolist() for weight in self._weights if partition_name in weight.indexed_name}
            if partition_weights_name == []:
                raise NameError(f"partition name {partition_name} is not exist, please check")
            else:
                return partition_weights_name

    @property
    def weights(self) -> dict:
        return {weight.indexed_name: weight.value.numpy().tolist() for weight in self._weights}

    @classmethod
    def release_loaded_weight(cls) -> NoReturn:
        del cls._restore_weights


