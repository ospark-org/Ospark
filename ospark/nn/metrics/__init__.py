import tensorflow as tf
from abc import ABC, abstractmethod
from typing import NoReturn

class Metrics(ABC):

    def __init__(self, class_category: dict) -> NoReturn:
        self.class_category  = class_category

    @abstractmethod
    def process(self, prediction, target) -> NoReturn:
        return NotImplemented

    @abstractmethod
    def get(self) -> dict:
        return NotImplemented
