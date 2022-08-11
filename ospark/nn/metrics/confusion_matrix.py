from . import Metrics
from typing import NoReturn
import numpy as np
import tensorflow as tf
from collections import defaultdict


class ConfusionMatrix(Metrics):

    def __init__(self, class_category: dict) -> NoReturn:
        self._class_category = class_category
        self._counter        = defaultdict(int)

    @property
    def counter(self):
        return self._counter

    @property
    def class_category(self) -> dict:
        return self._class_category

    def process(self, prediction: tf.Tensor, target: tf.Tensor) -> NoReturn:
        prediction_index = tf.argmax(prediction, axis=-1).numpy()
        target_index     = tf.argmax(target, axis=-1).numpy()
        for combined_index in zip(target_index[0], prediction_index[0]):
            self.counter[combined_index] += 1

    def calculate_start(self) -> dict:
        metrics = {}
        category = sorted(self.class_category.keys(), key=lambda x: self.class_category[x])
        basic_matrix = np.zeros(shape=[len(self.class_category), len(self.class_category)])
        for index, count in self.counter.items():
            basic_matrix[index] = count
        print(basic_matrix)
        metrics["accuracy"]  = np.sum(np.diagonal(basic_matrix)) / np.sum(basic_matrix)
        metrics["recall"]    = dict(zip(category, np.diagonal(basic_matrix) / np.sum(basic_matrix, axis=1)))
        metrics["precision"] = dict(zip(category, np.diagonal(basic_matrix) / np.sum(basic_matrix, axis=0)))
        return metrics
