from typing import NoReturn


class DataPath:

    def __init__(self, batch_size: int, total_data: int, index_table: list):
        self._batch_size      = batch_size
        self._total_data      = total_data
        self._index_table     = index_table

        self._training_data_paths = []
        self._label_data_paths    = []

    @property
    def training_data_paths(self):
        return self._training_data_paths

    @training_data_paths.setter
    def training_data_paths(self, path: str) -> NoReturn:
        self._training_data_paths.append(path)

    @property
    def label_data_paths(self):
        return self._label_data_paths

    @label_data_paths.setter
    def label_data_paths(self, path: str) -> NoReturn:
        self._label_data_paths.append(path)

    def range(self, step: int) -> range:
        self.reset()
        start_index = self._batch_size * step + 1
        end_index   = min(self._total_data, start_index + self._batch_size)
        return self._index_table[start_index: end_index]

    def reset(self) -> NoReturn:
        self._training_data_paths.clear()
        self._label_data_paths.clear()