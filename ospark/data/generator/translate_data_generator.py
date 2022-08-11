from __future__ import annotations
from typing import Optional, NoReturn, List, Tuple, Callable
import numpy as np
import tensorflow as tf
from . import DataGenerator
from ospark.data.encoder import LanguageDataEncoder
from tensorflow_datasets.core.deprecated.text.subword_text_encoder import SubwordTextEncoder
import math


class DataLength:

    def __init__(self):
        self._train_data_lens  = []
        self._target_data_lens = []

    @property
    def train_data_lens(self) -> list:
        return self._train_data_lens

    @property
    def target_data_lens(self) -> list:
        return self._target_data_lens

    @train_data_lens.setter
    def train_data_lens(self, datum_len: int):
        self._train_data_lens.append(datum_len)

    @target_data_lens.setter
    def target_data_lens(self, datum_len: int):
        self._target_data_lens.append(datum_len)

    @property
    def train_data_max_len(self) -> int:
        return max(self.train_data_lens)

    @property
    def target_data_max_len(self) -> int:
        return max(self.target_data_lens)

    def clear(self) -> NoReturn:
        self._train_data_lens.clear()
        self._target_data_lens.clear()


class TranslateDataGenerator(DataGenerator):

    class Dataset:

        def __init__(self):
            self._training_data = []
            self._target_data   = []
            self._data_lens     = DataLength()

        @property
        def training_data(self) -> tf.Tensor:
            return tf.concat([tf.pad(tf.convert_to_tensor(datum)[tf.newaxis, :], [[0, 0], [0, self.length.train_data_max_len - datum_len]]) for datum, datum_len
                              in zip(self._training_data, self.length.train_data_lens)], axis=0)

        @property
        def target_data(self) -> tf.Tensor:
            return tf.concat([tf.pad(tf.convert_to_tensor(datum)[tf.newaxis, :], [[0, 0], [0, self.length.target_data_max_len - datum_len]]) for datum, datum_len
                              in zip(self._target_data, self.length.target_data_lens)], axis=0)

        @training_data.setter
        def training_data(self, datum: tf.Tensor):
            self._training_data.append(datum)
            self._data_lens.train_data_lens = len(datum)

        @target_data.setter
        def target_data(self, datum: tf.Tensor):
            self._target_data.append(datum)
            self._data_lens.target_data_lens = len(datum)

        @property
        def batch_number(self) -> int:
            return len(self._training_data)

        @property
        def length(self) -> DataLength:
            return self._data_lens

        def clear(self) -> NoReturn:
            self._training_data.clear()
            self._target_data.clear()
            self._data_lens.clear()

    def __init__(self,
                 training_data: List[str],
                 target_data: List[str],
                 data_encoder: LanguageDataEncoder,
                 batch_size: int,
                 max_token: Optional[int]=None,
                 max_length: Optional[int]=None,
                 start_index: Optional[int]=None,
                 padding_range: Optional[int]=None) -> NoReturn:
        super().__init__(training_data=training_data, target_data=target_data, batch_size=batch_size, initial_step=0)
        self._data_number         = len(training_data)
        self._data_encoder        = data_encoder
        self._train_data_bos      = [data_encoder.train_data_encoder.vocab_size]  # 因為 vocab 中並沒有 bos 及 eos，所以新增額外兩個 index 當作 bos 及 eos
        self._target_data_bos     = [data_encoder.label_data_encoder.vocab_size]
        self._train_data_eos      = [data_encoder.train_data_encoder.vocab_size + 1]
        self._target_data_eos     = [data_encoder.label_data_encoder.vocab_size + 1]
        self._padding_range       = padding_range or 50
        self._max_length          = max_length
        self._next_interval       = None
        self._max_token           = max_token or 3000
        self._element_index       = start_index or 0
        self._get_data            = self.from_batch_limit if max_token is None else self.from_token_limit

    @property
    def data_number(self) -> int:
        return self._data_number

    @property
    def data_encoder(self) -> LanguageDataEncoder:
        return self._data_encoder

    @property
    def train_data_bos(self) -> List[int]:
        return self._train_data_bos

    @property
    def target_data_bos(self) -> List[int]:
        return self._target_data_bos

    @property
    def train_data_eos(self) -> List[int]:
        return self._train_data_eos

    @property
    def target_data_eos(self) -> List[int]:
        return self._target_data_eos

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def padding_range(self) -> int:
        return self._padding_range

    @property
    def max_token(self) -> int:
        return self._max_token

    @property
    def element_index(self) -> int:
        return self._element_index

    def encode_bos_eos(self,
                       train_sequence: str,
                       target_sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(train_sequence, tf.Tensor):
            train_sequence  = train_sequence.numpy().decode("utf-8")
            target_sequence = target_sequence.numpy().decode("utf-8")

        train_sequence = self.train_data_bos + self.data_encoder.encode_train_data(train_sequence) + self.train_data_eos
        target_sequence = self.target_data_bos + self.data_encoder.encode_label_data(target_sequence) + self.target_data_eos
        return np.array(train_sequence), np.array(target_sequence)

    def filter_length(self, train_sequence: np.ndarray, target_sequence: np.ndarray) -> bool:
        return len(train_sequence) > self.max_length or len(target_sequence) > self.max_length

    def from_batch_limit(self) -> Dataset:
        self.dataset.clear()
        while self.element_index < self.data_number:
            train_sequence, target_sequence = self.encode_bos_eos(train_sequence=self.training_data[self.element_index],
                                                                  target_sequence=self.target_data[self.element_index])
            if self.max_length is not None:
                is_filtered = self.filter_length(train_sequence=train_sequence, target_sequence=target_sequence)
                if is_filtered:
                    self._element_index += 1
                    continue
            self.dataset.training_data = train_sequence
            self.dataset.target_data   = target_sequence
            self._element_index += 1

            if self.batch_size == self.dataset.batch_number:
                break

        return self.dataset

    def from_token_limit(self) -> Dataset:
        padding_length = 0
        self.dataset.clear()
        while self.element_index < self.data_number:
            train_sequence, target_sequence = self.encode_bos_eos(train_sequence=self.training_data[self.element_index],
                                                                  target_sequence=self.target_data[self.element_index])
            if self.max_length is not None:
                is_filtered = self.filter_length(train_sequence=train_sequence, target_sequence=target_sequence)
                if is_filtered:
                    self._element_index += 1
                    continue

            self.dataset.training_data = train_sequence
            self.dataset.target_data   = target_sequence
            if self.padding_range is not None:
                padding_length = max(math.ceil(max(self.dataset.length.train_data_max_len, self.dataset.length.target_data_max_len) / self.padding_range) * self.padding_range,
                                     padding_length)
            else:
                padding_length = max(max(self.dataset.length.train_data_max_len, self.dataset.length.target_data_max_len), padding_length)

            token_number = padding_length * self.dataset.batch_number

            if self.max_token < token_number:
                self.dataset.length.train_data_lens.pop()
                self.dataset.length.target_data_lens.pop()
                self.dataset._training_data.pop()
                self.dataset._target_data.pop()
                break
            else:
                self._element_index += 1

        return self.dataset

    def __next__(self) -> Dataset:
        if self.element_index < self.data_number:
            return self._get_data()
        self.reset()
        raise StopIteration()

    def reset(self) -> NoReturn:
        self._element_index = 0

    def __call__(self):
        return self

    @classmethod
    def transform_into_tf_datasets(cls,
                                   train_data: List[str],
                                   target_data: List[str],
                                   data_encoder: LanguageDataEncoder,
                                   batch_size: int,
                                   max_token: Optional[int]=None,
                                   max_length: Optional[int]=None,
                                   start_index: Optional[int]=None,
                                   padding_range: Optional[int]=None) -> tf.data.Dataset:
        generator = cls(training_data=train_data,
                        target_data=target_data,
                        data_encoder=data_encoder,
                        batch_size=batch_size,
                        max_token=max_token,
                        max_length=max_length,
                        start_index=start_index,
                        padding_range=padding_range)
        generator = tf.data.Dataset.from_generator(generator,
                                                   output_signature=(tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                                                                     tf.TensorSpec(shape=(None, None), dtype=tf.int64)))
        return generator