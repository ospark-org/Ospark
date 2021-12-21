from ospark.nn.model import Model
from typing import NoReturn, Optional


class Predictor:

    def __init__(self, model: Model):
        self._model = model

    @property
    def model(self) -> Model:
        return self._model

    def predict(self, input_data):
        return NotImplementedError()

    def restore_weights(self, weights):
        return NotImplementedError()

    def __call__(self, input_data):
        prediction = self.predict(input_data=input_data)
        return prediction