from __future__ import annotations
from ospark.nn.component.basic_module import ModelObject, MetaObject
from typing import Optional, NoReturn, List, Tuple, Dict
from ospark import WeightOperator
from inspect import signature
from abc import abstractmethod
from functools import reduce
import tensorflow as tf
import numpy as np
import importlib


class ModelMeta(MetaObject):

    def __call__(self, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        if not obj._delay_create:
            obj.create()
            WeightOperator.clear_restore_weights()
            obj._trained_weights = None
        return obj


class Model(ModelObject, metaclass=ModelMeta):
    _weight_operator = WeightOperator()
    _show_info       = False

    def __init__(self,
                 obj_name: str,
                 delay_create: Optional[bool]=None,
                 is_training: Optional[bool]=None,
                 training_phase: Optional[bool]=None,
                 trained_weights: Optional[dict]=None):
        super(Model, self).__init__(obj_name=obj_name, is_training=is_training, training_phase=training_phase)
        self._delay_create    = delay_create if delay_create is not None else False
        self._trained_weights = trained_weights
        self._weight_operator.restore(weights=self._trained_weights or {})

    @property
    def training_weights(self) -> List[tf.Tensor]:
        weights = self._weight_operator.collect_weights(partition_name=self.obj_name)
        return weights

    def get_weights(self) -> dict:
        return self._weight_operator.weights

    # TODO: 目前沒辦法設定、修改 Weight 物件的 is_training。原因是 Weight 的資訊是寫在 class 中，所以不會變成 model info 傳出來，因此沒辦法修改
    #       方法一：在 create 時去針對個別權重修改，需要考量怎麼把資訊傳到 Operator
    #       方法二：提取 model info 時，也要可以拿到權重資訊，這部分的難度會更大，可能會需要整個重構才有辦法辦到。原因是因為 weight 是直接寫在 class 內的。
    #       測試目前的寫法是否正確。
    @classmethod
    def create_from_info(cls,
                         model_info: dict,
                         is_training: Optional[Dict[str, bool]]=None,
                         training_phase: Optional[Dict[str, bool]]=None,
                         trained_weights: Optional[dict]=None,
                         full_name: Optional[str]=None,
                         is_prediction_mode: Optional[bool]=None) -> Model:
        import_cls = getattr(importlib.import_module(model_info["import_path"]), model_info["class_name"])
        obj_name   = model_info["kwargs"].get("obj_name") or import_cls.__name__
        full_name  = f"{full_name}/{obj_name}" if full_name is not None else obj_name
        kwargs     = {}

        is_prediction_mode = is_prediction_mode if is_prediction_mode is not None else False

        for para_name, value in model_info["kwargs"].items():
            if type(value) == dict and value.get("class_name") is not None:
                value = cls.create_from_info(model_info=value,
                                             is_training=is_training,
                                             training_phase=training_phase,
                                             full_name=full_name,
                                             is_prediction_mode=is_prediction_mode)

            elif type(value) == list and type(value[0]) == dict:
                model_objs = []
                for _model_info in value:
                    model_objs.append(cls.create_from_info(model_info=_model_info,
                                                           is_training=is_training,
                                                           training_phase=training_phase,
                                                           full_name=full_name,
                                                           is_prediction_mode=is_prediction_mode))
                value = model_objs

            kwargs[para_name] = value

        if trained_weights is not None: kwargs["trained_weights"] = trained_weights

        if not is_prediction_mode:
            if is_training is not None and full_name in is_training:
                kwargs["is_training"] = is_training.pop(full_name)

            if training_phase is not None and full_name in training_phase:
                kwargs["training_phase"] = training_phase.pop(full_name)
        else:

            paras = list(signature(import_cls.__init__).parameters.keys())
            if "is_training" in paras:
                kwargs["is_training"] = False
            elif cls._show_info:
                print(f"{import_cls} not exits is_training", paras)

            if "training_phase" in paras:
                kwargs["training_phase"] = False
            elif cls._show_info:
                print(f"{import_cls} not exits training_phase", paras)
        try:
            model = import_cls(**kwargs)
        except TypeError as t:
            raise TypeError(f"{import_cls} {t}")
        return model

    @classmethod
    def present_model_architecture(cls, weights_name: iter) -> None:
        def reverse(weights: dict, names: List[str]):
            if len(names) != 0:
                name = names.pop(0)
                reverse(weights.setdefault(name, {}), names)

        def show_architecture(model_weights: dict, layer_depth: Optional[int]=None):
            layer_depth = layer_depth + 1 if layer_depth is not None else 0
            for layer_name, sub_layer in model_weights.items():

                suffix_word = f"-{layer_name}"

                if layer_depth == 0:
                    prefix_word = ""
                elif layer_depth == 1:
                    prefix_word = f"|"
                else:
                    prefix_word = f"|" + "|".join(["  " for i in range(layer_depth - 1)]) + "|"
                print(prefix_word + suffix_word)
                if sub_layer != {}:
                    show_architecture(sub_layer, layer_depth)

        weights = {}

        for name in weights_name:
            reverse(weights, name.split("/"))

        show_architecture(weights)

    def replace_weights(self, weights: dict) -> NoReturn:
        self._weight_operator.restore(weights=weights)
        self.create()
        self._weight_operator.clear_restore_weights()

    def __repr__(self):
        weights          = self._weight_operator.get_weights(partition_name=self.obj_name)
        parameter_number = 0
        for weight in weights.values():
            parameter_number += reduce(lambda init_value, shape: init_value * shape, np.array(weight).shape, 1)
        parameter_number = '{:,}'.format(parameter_number)
        return f"Parameters number: {parameter_number}"

    @abstractmethod
    def pipeline(self, *args, **kwargs) -> tf.Tensor:
        pass