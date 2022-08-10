from ospark.nn.component.basic_module import ModelObject, MetaObject
from typing import Optional, NoReturn, List
from ospark import WeightOperator
import tensorflow as tf


class ModelMeta(MetaObject):

    def __call__(self, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.create()
        WeightOperator.clean_weights()
        return obj


class Model(ModelObject, metaclass=ModelMeta):

    def __init__(self, obj_name: str, is_training: Optional[bool]=None, trained_weights: Optional[dict]=None):
        super(Model, self).__init__(obj_name=obj_name, is_training=is_training)
        self._weight_operator = WeightOperator()
        self._weight_operator.restore(weights=trained_weights or {})

    def replace_weights(self, weights: dict) -> NoReturn:
        self._weight_operator.restore(weights=weights)
        self.create()
        self._weight_operator.clean_weights()

    @property
    def training_weights(self) -> List[tf.Tensor]:
        weights = self._weight_operator.collect_weights(partition_name=self.obj_name)
        return weights

    def get_weights(self) -> dict:
        return self._weight_operator.weights

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()