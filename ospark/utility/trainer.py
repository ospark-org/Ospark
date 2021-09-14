from ospark.nn.component.weight import WeightOperator
from ospark.nn.loss_function import LossFunction
from ospark.nn.optimizer.learning_rate_schedule import LearningRateSchedule
from ospark.data.generator import DataGenerator
from typing import NoReturn, Union
from tensorflow.keras.optimizers import Optimizer, SGD


class Trainer:

    def __init__(self,
                 data_generator: DataGenerator,
                 batch_size: int,
                 epoch_number: int,
                 optimizer: Optimizer,
                 learning_rate: Union[float, LearningRateSchedule]):
        self._data_generator   = data_generator
        self._batch_size       = batch_size
        self._weights_operator = WeightOperator()
        self._epoch_number     = epoch_number
        self._optimizer        = optimizer(learning_rate=learning_rate)

    @property
    def weights_operator(self) -> WeightOperator:
        return self._weights_operator

    @property
    def data_generator(self) -> DataGenerator:
        return self._data_generator

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def epoch_number(self) -> int:
        return self._epoch_number

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer
    
    def start(self):
        return NotImplementedError()

    def get_weights(self) -> dict:
        return self.weights_operator.get

    def restore_weights(self, weight: dict={}) -> NoReturn:
        WeightOperator.restore(weight=weight)

