from __future__ import annotations

import random
from typing import List
from ospark.data.generator import DataGenerator
from typing import Generator, Tuple
from pydantic import BaseModel, Field
from PIL import Image
import math
import tensorflow as tf
import pathlib
import numpy as np


class DVAEDataGenerator:
    """
    DVAR data generator.
    """

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

        def setting_data(self, train_data: tf.Tensor, target_data: tf.Tensor) -> None:
            """
            Setting training data and target data.

            Args:
                train_data: tf.Tensor
                target_data: tf.Tensor
            """

            self._training_data = train_data
            self._target_data   = target_data

    def __init__(self,
                 train_data_folder: str,
                 batch_size: int,
                 resize_target: Tuple[int]):
        """
        Args:
            train_data_folder: str
            batch_size: int
            resize_target: Tuple[int]
        """

        self._train_data_folder = pathlib.Path(train_data_folder)
        self._files             = list(filter(lambda name: str(name).split("/")[-1][0] != "." and str(name)[-4:] == ".jpg", self._train_data_folder.iterdir()))
        self._files_number      = len(self._files)
        self._batch_size        = batch_size
        self._resize_target     = resize_target
        self._max_step          = math.ceil(self._files_number / batch_size)
        self._dataset           = self.Dataset()
        self._step              = 0

    @property
    def train_data_folder(self) -> str:
        return self._train_data_folder

    @property
    def files(self) -> Generator[pathlib.Path]:
        return self._files

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def max_step(self) -> int:
        return self._max_step

    def __iter__(self) -> DVAEDataGenerator:
        return self

    def __next__(self) -> Dataset:
        if self._step < self._max_step:
            self.setting_data()
            return self.dataset
        self._step = 0
        raise StopIteration()

    def setting_data(self) -> None:
        """
        Setting training data and target data.
        """

        start_point = self._step * self.batch_size
        end_point   = min(start_point + self.batch_size, self._files_number)
        images      = []
        images      = self.process_image(data_paths=self._files[start_point: end_point], images=images)

        if len(images) < self.batch_size:
            data_paths = random.sample(self._files, self.batch_size - len(images))
            images     = self.process_image(data_paths=data_paths, images=images)

        images = tf.convert_to_tensor(np.concatenate(images, axis=0).astype(np.float32))
        self.dataset.setting_data(train_data=images, target_data=images)
        self._step += 1

    def process_image(self, data_paths: List[pathlib.Path], images: List[Image]) -> Image:
        """
        Load and resize image.

        Args:
            data_paths: List[pathlib.Path]
            images: List[Image]

        Returns:
            images: Image
        """

        for data_path in data_paths:
            img = Image.open(data_path)
            if img.size != self._resize_target:
                img = img.resize(size=self._resize_target)
            images.append(np.array(img)[np.newaxis, :, :])
        return images