from typing import NoReturn


class SaveDelegate:

    def __init__(self):
        pass

    def save(self, save_obj: dict, path: str) -> NoReturn:
        raise NotImplementedError()