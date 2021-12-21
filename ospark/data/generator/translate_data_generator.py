from __future__ import annotations
from typing import Optional, NoReturn, List, Tuple, Callable
import numpy as np
import tensorflow as tf
from tensorflow_datasets.core.deprecated.text.subword_text_encoder import SubwordTextEncoder
import math


class TranslateDataGenerator:

    def __init__(self,
                 train_data: List[str],
                 target_data: List[str],
                 train_data_encoder: SubwordTextEncoder,
                 target_data_encoder: SubwordTextEncoder,
                 batch_size: int,
                 max_token: Optional[int]=None,
                 max_length: Optional[int]=None,
                 max_ratio: Optional[float]=None,
                 start_index: Optional[int]=None,
                 padding_range: Optional[int]=None) -> NoReturn:
        self._train_data          = train_data
        self._target_data         = target_data
        self._data_number         = len(train_data)
        self._batch_size          = batch_size
        self._train_data_encoder  = train_data_encoder
        self._target_data_encoder = target_data_encoder
        self._train_data_bos      = [train_data_encoder.vocab_size]  # 因為 vocab 中並沒有 bos 及 eos，所以新增額外兩個 index 當作 bos 及 eos
        self._target_data_bos     = [target_data_encoder.vocab_size]
        self._train_data_eos      = [train_data_encoder.vocab_size + 1]
        self._target_data_eos     = [target_data_encoder.vocab_size + 1]
        self._filter_rules        = []
        self._padding_range       = padding_range
        self._max_length          = max_length
        self._next_interval       = None
        if max_length is not None:
            self._filter_rules.append(self.filter_length)
        self._max_ratio = max_ratio
        if max_ratio is not None:
            self._filter_rules.append(self.filter_ratio)
        self._max_token           = max_token
        self._element_index       = start_index or 0
        self._padded_shape        = ([-1], [-1]) if max_length is None else ([max_length], [max_length])
        self._get_data            = self.batch_get_data if max_token is None else self.token_get_data

    @property
    def train_data(self) -> List[str]:
        return self._train_data

    @property
    def target_data(self) -> List[str]:
        return self._target_data

    @property
    def data_number(self) -> int:
        return self._data_number

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def train_data_encoder(self) -> SubwordTextEncoder:
        return self._train_data_encoder

    @property
    def target_data_encoder(self) -> SubwordTextEncoder:
        return self._target_data_encoder

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
    def max_ratio(self) -> float:
        return self._max_ratio

    @property
    def element_index(self) -> int:
        return self._element_index

    @property
    def filter_rules(self) -> list:
        return self._filter_rules

    def add_filter_rules(self, fn: Callable[[List[int], List[int]], bool]):
        self._filter_rules.append(fn)

    def encode_bos_eos(self,
                       train_sequence: str,
                       target_sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        train_sequence = self.train_data_bos + self.train_data_encoder.encode(train_sequence) + self.train_data_eos
        target_sequence = self.target_data_bos + self.target_data_encoder.encode(target_sequence) + self.target_data_eos
        return np.array(train_sequence), np.array(target_sequence)

    def filter_length(self, train_sequence: List[int], target_sequence: List[int]) -> bool:
        return all([len(train_sequence) <= self.max_length,
                    len(target_sequence) <= self.max_length])

    def filter_ratio(self, train_sequence: List[int], target_sequence: List[int]) -> bool:
        return all([len(train_sequence) / len(target_sequence) <= self.max_ratio,
                    len(target_sequence) / len(train_sequence) <= self.max_ratio])

    def batch_get_data(self):
        train_data = []
        target_data = []
        train_data_len = []
        target_data_len = []
        while self.element_index < self.data_number:
            train_sequence, target_sequence = self.encode_bos_eos(train_sequence=self.train_data[self.element_index],
                                                                  target_sequence=self.target_data[self.element_index])
            if self.filter_rules != []:
                result = [filter_method(train_sequence, target_sequence) for filter_method in self.filter_rules]
                if not all(result):
                    self._element_index += 1
                    continue

            train_data_len.append(len(train_sequence))
            target_data_len.append(len(target_sequence))
            train_data.append(train_sequence)
            target_data.append(target_sequence)
            self._element_index += 1
            if self.batch_size == len(train_data):
                break

        max_train_length = max(train_data_len)
        max_target_length = max(target_data_len)
        train_data = np.concatenate([np.pad(data, [0, max_train_length - length])[np.newaxis, :]
                                     for length, data in zip(train_data_len, train_data)])
        target_data = np.concatenate([np.pad(data, [0, max_target_length - length])[np.newaxis, :]
                                      for length, data in zip(target_data_len, target_data)])

        return tf.convert_to_tensor(train_data, dtype=tf.int64), \
               tf.convert_to_tensor(target_data, dtype=tf.int64)

    def token_get_data(self):
        train_data          = []
        target_data         = []
        train_data_len      = []
        target_data_len     = []
        last_padding_length = 0
        while self.element_index < self.data_number:
            train_sequence, target_sequence = self.encode_bos_eos(train_sequence=self.train_data[self.element_index],
                                                                  target_sequence=self.target_data[self.element_index])
            if self.filter_rules != []:
                result = [filter_method(train_sequence, target_sequence) for filter_method in self.filter_rules]
                if not all(result):
                    self._element_index += 1
                    continue

            train_data_len.append(len(train_sequence))
            target_data_len.append(len(target_sequence))
            train_data.append(train_sequence)
            target_data.append(target_sequence)
            if self.padding_range is not None:
                padding_length = max(math.ceil(max(len(train_sequence), len(target_sequence)) / self.padding_range) * self.padding_range,
                                     last_padding_length)
            else:
                padding_length = max(max(len(train_sequence), len(target_sequence)), last_padding_length)

            if last_padding_length == padding_length:
                self._next_interval = False
            else:
                self._next_interval = True

            token_number = padding_length * len(train_data)

            if self.max_token < token_number:
                train_data_len.pop()
                target_data_len.pop()
                train_data.pop()
                target_data.pop()
                if self._next_interval:
                    padding_length = last_padding_length
                if train_data != []:
                    break
                else:
                    self._element_index += 1
            else:
                self._element_index += 1
                last_padding_length  = padding_length
        train_data = np.concatenate([np.pad(data, [0, padding_length - length])[np.newaxis, :]
                                     for length, data in zip(train_data_len, train_data)])
        target_data = np.concatenate([np.pad(data, [0, padding_length - length])[np.newaxis, :]
                                      for length, data in zip(target_data_len, target_data)])

        return tf.convert_to_tensor(train_data, dtype=tf.int64), \
               tf.convert_to_tensor(target_data, dtype=tf.int64)

    def __iter__(self):
        return self

    def __next__(self):
        if self.element_index < self.data_number:
            train_data, target_data = self._get_data()
            return train_data, target_data
        self.reset()
        raise StopIteration()

    def reset(self) -> NoReturn:
        self._element_index = 0

    def __call__(self):
        return self

    @classmethod
    def wrapped_tf_datasets(cls,
                            train_data: List[str],
                            target_data: List[str],
                            train_data_encoder: SubwordTextEncoder,
                            target_data_encoder: SubwordTextEncoder,
                            batch_size: int,
                            max_token: Optional[int]=None,
                            max_length: Optional[int]=None,
                            max_ratio: Optional[float]=None,
                            start_index: Optional[int]=None,
                            padding_range: Optional[int]=None) -> tf.data.Dataset:
        generator = cls(train_data=train_data,
                        target_data=target_data,
                        train_data_encoder=train_data_encoder,
                        target_data_encoder=target_data_encoder,
                        batch_size=batch_size,
                        max_token=max_token,
                        max_length=max_length,
                        max_ratio=max_ratio,
                        start_index=start_index,
                        padding_range=padding_range)
        generator = tf.data.Dataset.from_generator(generator,
                                                   output_signature=(tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                                                                     tf.TensorSpec(shape=(None, None), dtype=tf.int64)))
        return generator