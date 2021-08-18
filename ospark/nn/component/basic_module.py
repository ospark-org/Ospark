from __future__ import annotations
from abc import ABC
from ospark.nn.component.weight import Weight
from typing import NoReturn, Optional


class BasicModule(ABC):

    def __init__(self, obj_name: str):
        self._obj_name = obj_name
        self._assigned = Assigned()

    @property
    def obj_name(self) -> str:
        return self._obj_name

    @property
    def assigned(self) -> Assigned:
        return self._assigned

    def initialize(self) -> NoReturn:
        pass

    def assign(self, component: BasicModule, name: Optional[str]=None) -> NoReturn:
        self.assigned.assign(component, name)

    def create(self, prefix_word: str) -> NoReturn:
        self.initialize()
        prefix_word += f"_{self.obj_name}"
        for component in self.assigned:
            component.create(prefix_word)


class Assigned:

    def __init__(self) -> NoReturn:
        self.component_names = []

    def assign(self, component: BasicModule, name: Optional[str]=None) -> NoReturn:
        if name is None:
            name = component.obj_name
        setattr(self, name, component)
        self.component_names.append(name)

    def __getattr__(self, name):
        return self.__dict__[name]

    def __getattribute__(self, name: str):
        obj = super().__getattribute__(name)
        if isinstance(obj, Weight):
            return obj.value
        else:
            return obj

    def __iter__(self) -> Assigned:
        return self

    def __next__(self):
        if self.component_names != []:
            obj_name = self.component_names.pop()
            obj = self.__getattr__(obj_name)
            return obj
        raise StopIteration