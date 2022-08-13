import json
from ospark.nn.model import Model
from  pathlib import Path


class DataOperator:

    @staticmethod
    def load(path: str) -> dict:
        raise NotImplementedError()

    @staticmethod
    def save(path: str, obj: dict) -> None:
        raise NotImplementedError()


class JsonOperator(DataOperator):

    @staticmethod
    def load(path: str) -> dict:
        with open(path, 'r') as fp:
            json_data = json.load(fp=fp)
        return json_data

    @staticmethod
    def save(folder_path: str, model: Model) -> None:
        save_path = Path(folder_path) / (model.obj_name + "_weights.json")
        with open(save_path, 'w') as fp:
            json.dump(obj=model.get_weights(), fp=fp)