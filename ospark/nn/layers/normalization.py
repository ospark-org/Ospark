import tensorflow as tf

import ospark.utility.weight_initializer
from . import Layer
from typing import NoReturn, List, Optional, Callable, Union
import ospark


class Normalization(Layer):

    def __init__(self,
                 obj_name: str,
                 layer_dimension: Union[int, list],
                 training_phase: Optional[bool]=None,
                 is_training: Optional[bool]=None,
                 gamma: Optional[float]=None,
                 beta: Optional[float]=None,
                 epsilon: Optional[float]=None,
                 use_bias: Optional[bool]=None,
                 use_scale: Optional[bool]=None):
        super().__init__(obj_name=obj_name, is_training=is_training, training_phase=training_phase)
        self._layer_dimension = layer_dimension

        self._gamma     = gamma or 1.0
        self._beta      = beta or 0.0
        self._use_bias  = use_bias if use_bias is not None else True
        self._use_scale = use_scale if use_scale is not None else True

        if self.use_scale:
            self._gamma = ospark.utility.weight_initializer.ones(obj_name="gamma", shape=self._layer_dimension) * tf.constant(self._gamma)
        if self.use_bias:
            self._beta = ospark.utility.weight_initializer.ones(obj_name="beta", shape=self._layer_dimension) * tf.constant(self._beta)

        self._epsilon = epsilon or 0.0001

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

    def __init_subclass__(cls) -> NoReturn:
        super().__init_subclass__()
        setattr(Normalization, cls.__name__, cls)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        return self.pipeline(input_data)


class PassNormalization(Normalization):

    def __init__(self):
        super(PassNormalization, self).__init__(obj_name="pass_norm",
                                                layer_dimension=0,
                                                is_training=False)

    def in_creating(self) -> NoReturn:
        pass

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
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

    def pipeline(self, input_data: tf.Tensor, axis: int=-1) -> tf.Tensor:
        mean, variance = tf.nn.moments(input_data, axes=[axis], keepdims=True)
        normalization_outputs = (input_data - mean) / (tf.sqrt(variance) + self.epsilon)
        if self.use_scale:
            normalization_outputs *= self._gamma
        if self.use_bias:
            normalization_outputs += self._beta
        return normalization_outputs


class BatchNormalization(Normalization):

    def __init__(self,
                 layer_dimension: Union[tf.Tensor, int],
                 training_phase: Optional[bool]=None,
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
                                                 layer_dimension=layer_dimension,
                                                 gamma=gamma,
                                                 beta=beta,
                                                 epsilon=epsilon,
                                                 use_scale=use_scale,
                                                 use_bias=use_bias,
                                                 training_phase=training_phase)

        moving_mean     = moving_mean or 0.0
        moving_variance = moving_variance or 1.0

        self._momentum        = momentum or 0.9
        self._moving_mean     = ospark.utility.weight_initializer.ones(obj_name="moving_mean",
                                                                       shape=[self._layer_dimension],
                                                                       trainable=False) * moving_mean
        self._moving_variance = ospark.utility.weight_initializer.ones(obj_name="moving_variance",
                                                                       shape=[self._layer_dimension],
                                                                       trainable=False) * moving_variance

    @property
    def momentum(self) -> tf.Tensor:
        return tf.constant(self._momentum)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        if self.training_phase:
            return self.train_process(input_data=input_data)
        else:
            return self.inference_process(input_data=input_data)

    def train_process(self, input_data: tf.Tensor) -> tf.Tensor:
        mean, variance  = tf.nn.moments(x=input_data, axes=[0, 1, 2])
        moving_mean     = self._moving_mean * self.momentum + mean * (tf.constant(1.) - self.momentum)
        self._moving_mean.assign(moving_mean)
        moving_variance = self._moving_variance * self.momentum + variance * (tf.constant(1.) - self.momentum)
        self._moving_variance.assign(moving_variance)
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

    def get_model_info(self) -> dict:
        BN_paras = super().get_model_info()
        print(BN_paras)
        return BN_paras