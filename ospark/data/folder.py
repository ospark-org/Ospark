from pathlib import Path
import numpy as np
from typing import Tuple


class DataFolder:

    def __init__(self,
                 train_data_folder: str,
                 label_data_folder: str):
        self._train_data_folder = Path(train_data_folder)
        self._label_data_folder = Path(label_data_folder)
        self._training_files    = list(self._train_data_folder.iterdir())
        self._labeling_files    = list(self._label_data_folder.iterdir())

        self._indexed_training_data = self.data_indexing(self.training_files)
        self._indexed_labeling_data = self.data_indexing(self.labeling_files)

    @property
    def train_data_folder(self) -> Path:
        return self._train_data_folder

    @property
    def label_data_folder(self) -> Path:
        return self._label_data_folder

    @property
    def training_files(self) -> list:
        return [str(file_name).split("/")[-1] for file_name in self._training_files]

    @property
    def labeling_files(self) -> list:
        return [str(file_name).split("/")[-1] for file_name in self._labeling_files]

    @property
    def indexed_training_data(self) -> dict:
        return self._indexed_training_data

    @property
    def indexed_labeling_data(self) -> dict:
        return self._indexed_labeling_data

    def data_indexing(self, files: list) -> dict:
        files.sort()

        filtered_file = [string for string in files if any([char.isdigit() for char in string])]
        datasets = map(self._indexed, filtered_file)
        return dict(datasets)

    def _indexed(self, file_name: str) -> Tuple[int, str]:
        number_index = np.where(np.array([char.isdigit() for char in file_name]) == True)[0]
        index        = int("".join([file_name[i] for i in number_index]))
        return index, file_name

    def get_files(self, index: int) -> Tuple[Path, Path]:
        training_data = self.train_data_folder / self.indexed_training_data[index]
        labeling_data = self.label_data_folder / self.indexed_labeling_data[index]
        return training_data, labeling_data