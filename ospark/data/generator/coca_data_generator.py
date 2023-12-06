from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Union
from functools import lru_cache
from PIL import Image
import pathlib
import tensorflow as tf
import ndjson
import math
import json
import numpy as np
import random


class CoCaDateGenerator:
    """
    CoCa data generator.
    """

    class Dataset:

        def __init__(self):
            self._train_data  = None
            self._target_data = None

        @property
        def training_data(self) -> Dict[str, tf.Tensor]:
            return self._train_data

        @property
        def target_data(self) -> tf.Tensor:
            return self._target_data

        def setting_data(self, training_data: Dict[str, tf.Tensor], target_data: tf.Tensor) -> None:
            """
            Setting training data and target data.

            Args:
                training_data: Dict[str, tf.Tensor]
                target_data: tf.Tensor
            """

            self._train_data  = training_data
            self._target_data = target_data

    def __init__(self,
                 training_dataset_path: str,
                 corpus_data_or_path: Union[str, dict],
                 batch_size: int,
                 resize_target: List[int]):
        """
        Args:
            training_dataset_path: str
            corpus_data_or_path: Union[str, dict]
            batch_size: int
            resize_target: List[int]
        """

        self._training_dataset_path = training_dataset_path
        self._batch_size            = batch_size
        self._resize_target         = resize_target
        self._dataset               = self.Dataset()
        self._step                  = 0
        self._max_step              = math.ceil(len(self.training_dataset) / batch_size)

        if type(corpus_data_or_path) == str:
            with open(corpus_data_or_path, 'r') as fp:
                corpus = json.load(fp)
            self._corpus = corpus
        else:
            self._corpus = corpus_data_or_path

    @property
    @lru_cache(maxsize=128)
    def training_dataset(self) -> List[dict]:
        with open(self._training_dataset_path, 'r') as fp:
            dataset = ndjson.load(fp)
        return dataset

    @property
    def corpus(self) -> dict:
        return self._corpus

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_step(self) -> int:
        return self._max_step

    @property
    def random_index(self) -> int:
        return random.randint(0, len(self.training_dataset) - 1)

    def __iter__(self):
        return self

    def __next__(self):
        if self._step < self._max_step:
            dataset = self.get_data()

            self._step += 1
            return dataset
        else:
            self._step = 0
            raise StopIteration()

    def get_data(self) -> Dataset:
        """
        Returns:
            datasets: Dataset
        """

        start_point = self._step * self.batch_size
        end_point   = start_point + self.batch_size
        images      = []
        text_data   = []
        for index in range(start_point, min(self.max_step, end_point)):
            images, text_data = self.load_data(datum_index=index, images=images, text_data=text_data)

        while len(text_data) != self.batch_size:
            images, text_data = self.load_data(datum_index=self.random_index, images=images, text_data=text_data)

        input_sequence, target_sequence = self.create_text_target(text=text_data)
        input_sequence   = tf.convert_to_tensor(input_sequence, dtype=tf.float32)
        target_data      = tf.convert_to_tensor(target_sequence, dtype=tf.float32)
        image_data       = tf.convert_to_tensor(np.concatenate(images, axis=0), dtype=tf.float32)
        training_data    = {"image": image_data, "text": input_sequence}
        self._dataset.setting_data(training_data=training_data, target_data=target_data)
        return self._dataset

    def load_data(self, datum_index: int, images: List[Image.Image], text_data: List[str]) -> Tuple[List[Image.Image],  List[str]]:
        """
        Load data.

        Args:
            datum_index: int
            images: List[Image.Image]
            text_data: List[str]

        Returns:
            images: List[Image.Image]
            text_data: List[str]
        """

        data = self.training_dataset[datum_index]

        image_path   = data["image_path"]
        image_path   = pathlib.Path(image_path)
        image        = self.process_image(data_path=image_path)
        images.append(image)

        text = data["target_data"]
        text_data.append(text)
        return images, text_data

    def process_image(self, data_path: pathlib.Path) -> Image:
        """
        Load and resize image.

        Args:
            data_path: pathlib.Path

        Returns:
            img: Image
        """

        img = Image.open(data_path)
        if img.size != self._resize_target:
            img = img.resize(size=self._resize_target)
        img = np.array(img)[np.newaxis, :, :]
        return img

    def create_text_target(self, text: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create text target.

        Args:
            text: List[str]

        Returns:
            result: Tuple[np.ndarray, np.ndarray]
        """

        lengths     = [len(text_sequence) for text_sequence in text]
        max_length  = max(lengths) + 1 # Because add BOS and EOS
        input_data  = []
        target_data = []
        for i, text_sequence in enumerate(text):
            char_index  = [self.corpus[char] if char in self.corpus else self._corpus.setdefault(char, len(self._corpus)) for char in text_sequence]

            input_text  = np.array([[self.corpus["BOS"]] + char_index])
            input_text  = np.pad(input_text, ((0, 0), (0, max_length - lengths[i])), constant_values=0)
            input_data.append(input_text)

            target_text = np.array([char_index + [self.corpus["EOS"]]])
            target_text = np.pad(target_text, ((0, 0), (0, max_length - lengths[i])), constant_values=0)
            target_data.append(target_text)
        return np.concatenate(input_data, axis=0), np.concatenate(target_data, axis=0)