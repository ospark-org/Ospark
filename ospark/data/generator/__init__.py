from __future__ import annotations
from typing import Tuple, Optional, Union
import tensorflow as tf
import math


class DataGenerator:

    class Dataset:

        def __init__(self):
            self._training_data = None
            self._target_data   = None

        @property
        def training_data(self) -> tf.Tensor:
            return self._training_data

        @property
        def target_data(self) -> tf.Tensor:
            return self._target_data

        def data_setting(self, training_data: tf.Tensor, target_data: tf.Tensor):
            self._training_data = training_data
            self._target_data   = target_data


    def __init__(self,
                 training_data: Union[tf.Tensor, list],
                 target_data: Union[tf.Tensor, list],
                 batch_size: int,
                 initial_step: Optional[int]=None):
        self._training_data  = training_data
        self._target_data    = target_data
        self._batch_size     = batch_size
        self._max_step       = math.ceil(len(training_data) / batch_size)
        self._step           = initial_step or 0
        self._dataset        = self.Dataset()

    @property
    def training_data(self) -> tf.Tensor:
        return self._training_data

    @property
    def target_data(self) -> tf.Tensor:
        return self._target_data

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def step(self) -> int:
        return self._step

    @property
    def max_step(self) -> int:
        return self._max_step

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def __iter__(self):
        return self

    def __next__(self) -> Dataset:
        if self.step < self.max_step:
            dataset = self._get_data()
            self._step += 1
            return dataset
        self.reset()
        raise StopIteration()

    def reset(self):
        self._step = 0

    def _get_data(self) -> Dataset:
        start = self.step * self.batch_size
        end   = start + self.batch_size

        training_data = self.training_data[start: end]
        target_data   = self.target_data[start: end]

        self.dataset.data_setting(training_data=training_data, target_data=target_data)
        return self.dataset

            