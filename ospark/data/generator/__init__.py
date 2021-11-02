from typing import List, Optional
import math

class DataGenerator:

    def __init__(self, training_data: List[str], target_data: List[str], batch_size: int, initial_step: Optional[int]=None):
        self._training_data  = training_data
        self._target_data    = target_data
        self._batch_size     = batch_size
        self._max_step       = math.ceil(len(training_data) / batch_size)
        self._step           = initial_step or 0

    @property
    def training_data(self) -> List[str]:
        return self._training_data

    @property
    def target_data(self) -> List[str]:
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

    def __iter__(self):
        return self

    def __next__(self):
        return NotImplementedError()

    def get_data(self):
        return NotImplementedError()
    
    def reset(self):
        self._step = 0

            