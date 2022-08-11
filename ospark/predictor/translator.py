from ospark.predictor import *
from ospark.models.former import Former
from typing import List
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


class Translator(Predictor):

    def __init__(self,
                 model: Former,
                 input_text_encoder: tfds.deprecated.text.SubwordTextEncoder,
                 output_text_encoder: tfds.deprecated.text.SubwordTextEncoder,
                 predict_max_length: int):
        super().__init__(model=model)
        self._input_text_encoder  = input_text_encoder
        self._output_text_encoder = output_text_encoder
        self._predict_max_length  = predict_max_length
        self._input_bos           = [input_text_encoder.vocab_size]
        self._output_bos          = [output_text_encoder.vocab_size]
        self._input_eos           = [input_text_encoder.vocab_size + 1]
        self._output_eos          = [output_text_encoder.vocab_size + 1]

    @property
    def input_text_encoder(self) -> tfds.deprecated.text.SubwordTextEncoder:
        return self._input_text_encoder

    @property
    def output_text_encoder(self) -> tfds.deprecated.text.SubwordTextEncoder:
        return self._output_text_encoder

    @property
    def predict_max_length(self) -> int:
        return self._predict_max_length

    @property
    def input_bos(self) -> List[int]:
        return self._input_bos

    @property
    def input_eos(self) -> List[int]:
        return self._input_eos

    @property
    def output_bos(self) -> tf.Tensor:
        return tf.convert_to_tensor(self._output_bos, dtype=tf.int32)[tf.newaxis, :]

    @property
    def output_eos(self) -> np.ndarray:
        return np.array(self._output_eos)

    def predict(self, input_data: str):
        input_encoded_text = tf.convert_to_tensor(self.input_bos + \
                                                  self.input_text_encoder.encode(input_data) + \
                                                  self.input_eos, dtype=tf.int32)[tf.newaxis, :]
        decoder_input      = self.output_bos
        for i in range(self.predict_max_length):
            prediction = self.model(encoder_input=input_encoded_text, decoder_input=decoder_input)
            prediction = tf.cast(tf.argmax(prediction, axis=-1), dtype=tf.int32)
            if prediction.numpy()[0, -1] == self.output_eos[0]:
                break
            decoder_input = tf.concat([self.output_bos, prediction], axis=-1)
        text_output = self.output_text_encoder.decode(decoder_input.numpy()[0, 1:])
        return text_output