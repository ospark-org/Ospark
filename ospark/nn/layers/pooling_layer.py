from ospark.nn.layers import Layer, NoReturn
from typing import List
import tensorflow as tf


class PoolingLayer(Layer):

    def __init_subclass__(cls) -> NoReturn:
        super().__init_subclass__()
        setattr(PoolingLayer, cls.__name__, cls)

    def __init__(self,
                 obj_name: str,
                 pooling_size: List[int],
                 strides: List[int],
                 padding: str):
        super().__init__(obj_name=obj_name)
        self._pooling_size = pooling_size
        self._strides      = strides
        self._padding      = padding

    @property
    def pooling_size(self) -> List[int]:
        return self._pooling_size

    @property
    def padding(self) -> str:
        return self._padding

    @property
    def strides(self) -> List[int]:
        return self._strides

    def pipeline(self, input_data: tf.Tensor):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)


class Max2DPooling(PoolingLayer):

    def __init__(self,
                 pooling_size: List[int],
                 strides: List[int],
                 padding: str):
        super().__init__(obj_name="max_2d",
                         pooling_size=pooling_size,
                         strides=strides,
                         padding=padding)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        pooling_result = tf.nn.max_pool2d(input=input_data,
                                          ksize=self.pooling_size,
                                          strides=self.strides,
                                          padding=self.padding)
        return pooling_result


class Average2DPooling(PoolingLayer):

    def __init__(self,
                 pooling_size: List[int],
                 strides: List[int],
                 padding: str):
        super().__init__(obj_name="average_2d",
                         pooling_size=pooling_size,
                         strides=strides,
                         padding=padding)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        pooling_result = tf.nn.avg_pool2d(input=input_data,
                                          ksize=self.pooling_size,
                                          strides=self.strides,
                                          padding=self.padding)
        return pooling_result