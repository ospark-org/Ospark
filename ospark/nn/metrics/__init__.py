from typing import NoReturn


class Metrics:

    def __init__(self) -> NoReturn:
        pass

    def process(self, prediction, target) -> NoReturn:
        return NotImplementedError()

    def calculate_start(self):
        return NotImplementedError()
