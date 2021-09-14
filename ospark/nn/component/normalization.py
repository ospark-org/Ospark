import tensorflow as tf
from ospark.nn.component.basic_module import BasicModule
from typing import NoReturn, List, Optional, Callable
import ospark

class Normalization(BasicModule):

    def __init__(self,
                 obj_name: str,
                 gamma: Optional[float]=None,
                 beta: Optional[float]=None,
                 epsilon: Optional[float]=None):
        super().__init__(obj_name=obj_name)
        # gamme 不是 function 所以不會是 initializer
        self._gamma   = gamma or 1.
        self._beta    = beta or 0.
        self._epsilon = epsilon or 0.0001

    @property
    def gamma(self) -> tf.Tensor:
        return self._gamma

    @property
    def beta(self) -> tf.Tensor:
        return self._beta

    @property
    def epsilon(self) -> tf.Tensor:
        return tf.constant(self._epsilon)

    def on_creating(self) -> NoReturn:
        self.assign(component=ospark.Weight(obj_name="gamma", initial_value=self.gamma))
        self.assign(component=ospark.Weight(obj_name="beta", initial_value=self.beta))

    def __init_subclass__(cls) -> NoReturn:
        super().__init_subclass__()
        setattr(Normalization, cls.__name__, cls)

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        return self.calculate(input_data)


class PassNormalization(Normalization):

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        return input_data


class LayerNormalization(Normalization):

    def __init__(self,
                 obj_name: Optional[str]=None,
                 gamma: Optional[float]=None,
                 beta: Optional[float]=None,
                 epsilon: Optional[float]=None):
        super().__init__(obj_name=obj_name or "layer_norm",
                         gamma=gamma,
                         beta=beta,
                         epsilon=epsilon)

    def calculate(self, input_data: tf.Tensor, axis: int=-1) -> tf.Tensor:
        mean, variance = tf.nn.moments(input_data, axes=[axis], keepdims=True)
        normalization_outputs = (input_data - mean) / (tf.sqrt(variance) + self.epsilon)
        return self.assigned.gamma * normalization_outputs + self.assigned.beta


class BatchNormalization(Normalization):

    def __init__(self,
                 input_depth: tf.Tensor,
                 obj_name: str=None,
                 gamma: Optional[float]=None,
                 beta: Optional[float]=None,
                 epsilon: Optional[float]=None,
                 momentum: Optional[float]=None,
                 moving_mean: Optional[float]=None,
                 moving_variance: Optional[float]=None,
                 trainable: bool=True):
        super(BatchNormalization, self).__init__(obj_name=obj_name or "batch_norm",
                                                 gamma=gamma,
                                                 beta=beta,
                                                 epsilon=epsilon)
        self._input_depth     = input_depth
        self._momentum        = momentum or 0.9
        self._moving_mean     = moving_mean or 0.
        self._moving_variance = moving_variance or 1.
        self._calculate       = self.train_process if trainable else self.inference_process

    @property
    def input_depth(self) -> tf.Tensor:
        return self._input_depth

    @property
    def momentum(self) -> tf.Tensor:
        return tf.constant(self._momentum)

    @property
    def moving_mean(self) -> tf.Tensor:
        return tf.ones(shape=[self.input_depth]) * self._moving_mean

    @property
    def moving_variance(self) -> tf.Tensor:
        return tf.ones(shape=[self.input_depth]) * self._moving_variance

    def calculate(self, input_data: tf.Tensor) -> Callable[[tf.Tensor], tf.Tensor]:
        return self._calculate(input_data)

    def on_creating(self) -> NoReturn:
        super().on_creating()
        self.assign(component=ospark.Weight(obj_name="moving_mean", initial_value=self.moving_mean))
        self.assign(component=ospark.Weight(obj_name="moving_variance", initial_value=self.moving_variance))

    def train_process(self, input_data: tf.Tensor) -> tf.Tensor:
        mean, variance = tf.nn.moments(x=input_data, axes=[0, 1, 2])
        moving_mean = self.assigned.moving_mean * self.momentum + mean * (1 - self.momentum)
        moving_varience = self.assigned.moving_variance * self.momentum + variance * (1 - self.momentum)
        return tf.nn.batch_normalization(x=input_data,
                                         mean=moving_mean,
                                         variance=moving_varience,
                                         scale=self.assigned.gamma,
                                         offset=self.assigned.beta,
                                         variance_epsilon=self.epsilon)

    def inference_process(self, input_data: tf.Tensor) -> tf.Tensor:
        return tf.nn.batch_normalization(x=input_data,
                                         mean=self.assigned.moving_mean,
                                         variance=self.assigned.moving_variance,
                                         scale=self.assigned.gamma,
                                         offset=self.assigned.beta,
                                         variance_epsilon=self.epsilon)