import tensorflow as tf

import ospark.utility.weight_initializer
from . import Layer
from typing import NoReturn, List, Optional, Callable, Union
import ospark


class Normalization(Layer):

    def __init__(self,
                 obj_name: str,
                 layer_dimension: Union[int, list],
                 is_training: Optional[bool]=None,
                 gamma: Optional[float]=None,
                 beta: Optional[float]=None,
                 epsilon: Optional[float]=None,
                 use_bias: Optional[bool]=None,
                 use_scale: Optional[bool]=None):
        super().__init__(obj_name=obj_name, is_training=is_training)
        self._layer_dimension = layer_dimension

        self._gamma     = gamma or 1.0
        self._beta      = beta or 0.0
        self._epsilon   = epsilon or 0.0001
        self._use_bias  = use_bias or True
        self._use_scale = use_scale or True

    @property
    def gamma(self) -> Union[float, ospark.Weight]:
        return self._gamma

    @property
    def beta(self) -> Union[float, ospark.Weight]:
        return self._beta

    @property
    def use_bias(self) -> bool:
        return self._use_bias

    @property
    def use_scale(self) -> bool:
        return self._use_scale

    @property
    def epsilon(self) -> tf.Tensor:
        return tf.constant(self._epsilon)

    def in_creating(self) -> NoReturn:
        if self.use_scale:
            self._gamma = ospark.utility.weight_initializer.ones(obj_name="gamma", shape=self._layer_dimension) * self._gamma
        if self.use_bias:
            self._beta = ospark.utility.weight_initializer.ones(obj_name="beta", shape=self._layer_dimension) * self._beta

    def __init_subclass__(cls) -> NoReturn:
        super().__init_subclass__()
        setattr(Normalization, cls.__name__, cls)

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        return self.calculate(input_data)


class PassNormalization(Normalization):

    def __init__(self):
        super(PassNormalization, self).__init__(obj_name="pass_norm",
                                                layer_dimension=0,
                                                is_training=False)

    def in_creating(self) -> NoReturn:
        pass

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        return input_data


class LayerNormalization(Normalization):

    def __init__(self,
                 layer_dimension: Union[int, list],
                 is_training: Optional[bool]=None,
                 obj_name: Optional[str]=None,
                 gamma: Optional[float]=None,
                 beta: Optional[float]=None,
                 epsilon: Optional[float]=None,
                 use_bias: Optional[bool]=True,
                 use_scale: Optional[bool]=True):
        super().__init__(obj_name=obj_name or "layer_norm",
                         is_training=is_training,
                         layer_dimension=layer_dimension,
                         gamma=gamma,
                         beta=beta,
                         epsilon=epsilon,
                         use_bias=use_bias,
                         use_scale=use_scale)

    def calculate(self, input_data: tf.Tensor, axis: int=-1) -> tf.Tensor:
        mean, variance = tf.nn.moments(input_data, axes=[axis], keepdims=True)
        normalization_outputs = (input_data - mean) / (tf.sqrt(variance) + self.epsilon)
        if self.use_scale:
            normalization_outputs *= self._gamma
        if self.use_bias:
            normalization_outputs += self._beta
        return normalization_outputs


class BatchNormalization(Normalization):

    def __init__(self,
                 input_depth: Union[tf.Tensor, int],
                 is_training: Optional[bool]=None,
                 obj_name: str=None,
                 gamma: Optional[float]=None,
                 beta: Optional[float]=None,
                 epsilon: Optional[float]=None,
                 momentum: Optional[float]=None,
                 moving_mean: Optional[float]=None,
                 moving_variance: Optional[float]=None,
                 use_bias: Optional[bool]=True,
                 use_scale: Optional[bool]=True):
        super(BatchNormalization, self).__init__(obj_name=obj_name or "batch_norm",
                                                 is_training=is_training,
                                                 layer_dimension=input_depth,
                                                 gamma=gamma,
                                                 beta=beta,
                                                 epsilon=epsilon,
                                                 use_scale=use_scale,
                                                 use_bias=use_bias)
        self._input_depth     = input_depth
        self._momentum        = momentum or 0.9
        self._moving_mean     = moving_mean or 0.0
        self._moving_variance = moving_variance or 1.0
        self._calculate       = self.train_process if is_training else self.inference_process

    @property
    def input_depth(self) -> Union[tf.Tensor, int]:
        return self._input_depth

    @property
    def momentum(self) -> tf.Tensor:
        return tf.constant(self._momentum)

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        return self._calculate(input_data)

    def in_creating(self) -> NoReturn:
        super().in_creating()
        self.assign(component=ospark.utility.weight_initializer.ones(obj_name="moving_mean",
                                                                     shape=[self.input_depth],
                                                                     trainable=False) * self._moving_mean)
        self.assign(component=ospark.utility.weight_initializer.ones(obj_name="moving_variance",
                                                                     shape=[self.input_depth],
                                                                     trainable=False) * self._moving_variance)

    def train_process(self, input_data: tf.Tensor) -> tf.Tensor:
        mean, variance  = tf.nn.moments(x=input_data, axes=[0, 1, 2])
        moving_mean     = self.assigned.moving_mean * self.momentum + mean * (1 - self.momentum)
        self.assigned.moving_mean.assign(moving_mean)
        moving_variance = self.assigned.moving_variance * self.momentum + variance * (1 - self.momentum)
        self.assigned.moving_variance.assign(moving_variance)
        normalization_outputs = tf.nn.batch_normalization(x=input_data,
                                                          mean=mean,
                                                          variance=variance,
                                                          offset=None,
                                                          scale=None,
                                                          variance_epsilon=self.epsilon)
        if self.use_scale:
            normalization_outputs *= self._gamma
        if self.use_bias:
            normalization_outputs += self._beta
        return normalization_outputs

    def inference_process(self, input_data: tf.Tensor) -> tf.Tensor:
        normalization_outputs = tf.nn.batch_normalization(x=input_data,
                                                          mean=self.assigned.moving_mean,
                                                          variance=self.assigned.moving_variance,
                                                          offset=None,
                                                          scale=None,
                                                          variance_epsilon=self.epsilon)
        if self.use_scale:
            normalization_outputs *= self._gamma
        if self.use_bias:
            normalization_outputs += self._beta
        return normalization_outputs