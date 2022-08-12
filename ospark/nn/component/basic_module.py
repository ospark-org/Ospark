from __future__ import annotations
from ospark.nn.component.weight import Weight
from typing import NoReturn, Optional, Set, Union, Any
from .name_space import NameSpace
import tensorflow as tf


# TODO 如果這邊 input_data 內有 Weight 的話，會有問題，因為沒辦法用 Descriptor 去 get
def dismantling_nest_structure(input_data: list, obj: ModelObject):
    for i, inside_obj in enumerate(input_data):
        if isinstance(inside_obj, ModelObject) or isinstance(inside_obj, Weight):
            obj.assign(inside_obj)
        elif isinstance(inside_obj, list):
            dismantling_nest_structure(input_data=inside_obj, obj=obj)
        else:
            continue


class Descriptor:

    def __init__(self):
        self._obj_package = {}

    def __get__(self, instance: Any, instance_type: Any) -> Union[tf.Tensor, ModelObject]:
        identifier = id(instance)
        if isinstance(self._obj_package[identifier], Weight):
            return self._obj_package[identifier].value
        else:
            return self._obj_package[identifier]

    def __set__(self, instance: ModelObject, value: Any) -> NoReturn:
        if isinstance(value, ModelObject) or isinstance(value, Weight):
            instance.assign(value)
            if isinstance(value, Weight) and value._is_default_setting:
                value._trainable = instance.is_training
            elif isinstance(value, ModelObject):
                value._is_training = instance.is_training
        identifier = id(instance)
        self._obj_package[identifier] = value


class MetaObject(type):

    cls = None

    def __new__(mcls, *args, **kwargs):
        cls = super().__new__(mcls, *args, **kwargs)
        if mcls.cls is None:
            mcls.cls = cls
        return cls

    def __call__(mcls, *args, **kwargs) -> ModelObject:
        obj = super().__call__(*args, **kwargs)
        obj.in_creating()

        keys = list(obj.__dict__.keys())
        for attr_name in keys:
            attr_value = obj.__dict__[attr_name]
            if isinstance(attr_value, ModelObject) or isinstance(attr_value, Weight):
                sub_obj = obj.__dict__.pop(attr_name)
                setattr(mcls.cls, attr_name, Descriptor())
                mcls.cls.__dict__[attr_name].__set__(obj, sub_obj)
            elif isinstance(attr_value, list):
                dismantling_nest_structure(input_data=attr_value, obj=obj)
        return obj


class ModelObject(metaclass=MetaObject):

    def __init__(self, obj_name: str, is_training: Optional[bool]=None):
        self._obj_name    = obj_name
        self._assigned    = Assigned()
        if is_training is None:
            self._is_training = True
        else:
            self._is_training = is_training

    @property
    def obj_name(self) -> str:
        return self._obj_name

    @property
    def assigned(self) -> Assigned:
        return self._assigned

    @property
    def is_training(self) -> bool:
        return self._is_training

    def in_creating(self) -> NoReturn:
        pass

    def assign(self, component: Union[ModelObject, Weight], name: Optional[str]=None) -> Union[ModelObject, Weight]:
        self.assigned.assign(component, name)
        return component

    def create(self) -> NoReturn:
        with NameSpace(name=self.obj_name):
            for component in self.assigned:
                component.create()


class Assigned:

    def __init__(self) -> NoReturn:
        self._component_names = set()

    @property
    def component_names(self) -> Set[str]:
        return self._component_names

    def assign(self, component: ModelObject, name: Optional[str]=None) -> NoReturn:
        if name is None:
            name = component.obj_name
        setattr(self, name, component)
        self._component_names.add(name)

    def __getattr__(self, name):
        if name != "is_tensor_like":
            return self.__dict__[name]

    def __getattribute__(self, name: str):
        obj = super().__getattribute__(name)
        if isinstance(obj, Weight):
            return obj.value
        else:
            return obj

    def __iter__(self) -> ModelObject:
        for component_name in self.component_names:
            obj = self.__getattr__(component_name)
            yield obj