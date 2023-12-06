from ospark.nn.component.weight import WeightOperator
from ospark.data.generator import DataGenerator
from ospark.data.save_delege import SaveDelegate
from typing import NoReturn, Optional, Callable, List, Tuple, Union, Dict
from tensorflow.keras.optimizers import Optimizer
from ospark.nn.loss_function.loss_function import LossFunction
from ospark.nn.model import Model
from logging import Logger
import tensorflow as tf
import logging
import json
import time


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

class Trainer:

    def __init__(self,
                 model: Model,
                 data_generator: DataGenerator,
                 epoch_number: int,
                 optimizer: Optimizer,
                 loss_function: Union[LossFunction, Dict[str, LossFunction]],
                 save_weights_path: str,
                 save_info_path: str,
                 presentation_of_loss_value: Optional[int]=None,
                 save_delegate: Optional[SaveDelegate]=None,
                 save_times: Optional[int]=None,
                 use_auto_graph: Optional[bool]=True,
                 use_multi_gpu: Optional[bool]=None,
                 devices: Optional[List[str]]=None,
                 logger: Optional[Union[Logger, str]]=None):
        """
        Args:
            model:
            data_generator:
            epoch_number:
            optimizer:
            loss_function:
            save_weights_path:
            save_info_path:
            presentation_of_loss_value:
            save_delegate:
            save_times:
            use_auto_graph:
            use_multi_gpu:
            devices:
            logger:
        """

        self._weights_operator  = WeightOperator()
        self._loss_function     = loss_function
        self._epoch_number      = epoch_number
        self._save_delegate     = save_delegate or self
        self._save_times        = save_times
        self._save_weights_path = save_weights_path
        self._save_info_path    = save_info_path

        if type(logger) == str:
            self._logger = logging.getLogger(name=logger)
        else:
            self._logger         = logger or logging

        self._presentation_of_loss_value = presentation_of_loss_value

        use_multi_gpu = use_multi_gpu or False
        if use_multi_gpu:
            self._mirrored_strategy = tf.distribute.MirroredStrategy() if devices is None \
                                      else tf.distribute.MirroredStrategy(devices=devices)

            with self.mirrored_strategy.scope():
                self._model     = model
                self._optimizer = optimizer

            self._data_generator  = self.mirrored_strategy.experimental_distribute_dataset(data_generator)
            self._training_method = self.distributed_train_step

        else:
            self._model            = model
            self._optimizer        = optimizer
            self._data_generator   = data_generator
            self._training_method  = self.graph_mode if use_auto_graph else self.eager_mode

    @property
    def model(self) -> Model:
        return self._model

    @property
    def loss_function(self) -> LossFunction:
        return self._loss_function

    @property
    def mirrored_strategy(self) -> tf.distribute.MirroredStrategy:
        return self._mirrored_strategy

    @property
    def weights_operator(self) -> WeightOperator:
        return self._weights_operator

    @property
    def data_generator(self) -> DataGenerator:
        return self._data_generator

    @property
    def epoch_number(self) -> int:
        return self._epoch_number

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def save_delegate(self) -> SaveDelegate:
        return self._save_delegate

    @property
    def save_times(self) -> int:
        return self._save_times

    @property
    def save_weights_path(self) -> str:
        return self._save_weights_path

    @property
    def training_method(self) -> Callable:
        return self._training_method

    @property
    def save_info_path(self) -> str:
        return self._save_info_path

    @property
    def logger(self) -> Logger:
        return self._logger

    def start(self):
        self._logger.info("=" * 24)
        self._logger.info("Training start.")
        self.training_process()
        self._logger.info("Training end.")
        self._logger.info("=" * 24)

    def training_process(self) -> NoReturn:
        for epoch in range(self.epoch_number):
            total_loss_value = 0
            training_count   = 0
            start_time       = time.time()
            for step, dataset in enumerate(self.data_generator):
                training_data, target_data = dataset.training_data, dataset.target_data
                loss_value = self.training_method(training_data, target_data)
                total_loss_value += loss_value
                training_count   += 1
                if self._presentation_of_loss_value is not None and step % self._presentation_of_loss_value == 0:
                    self._logger.info(f"step: {step}, loss value : {total_loss_value / training_count}")
                    self._logger.info("estimated time pre epoch: ", self.data_generator.max_step / (step + 1) * (time.time() - start_time))


            self._logger.info(f'Epoch {epoch + 1}, '
                              f'Loss {total_loss_value / training_count:.4f}')
            self._logger.info(f'Time taken for 1 epoch: {time.time() - start_time:.2f} secs\n')
            if self.will_save(epoch_number=epoch) and self.save_weights_path is not None:
                self.save_delegate.save(save_obj=self.weights_operator.weights, path=self.save_weights_path)
                self.save_delegate.save(save_obj=self.model.get_model_info(), path=self.save_info_path)

        if self.save_weights_path is not None:
            self.save_delegate.save(save_obj=self.weights_operator.weights, path=self.save_weights_path)
            self.save_delegate.save(save_obj=self.model.get_model_info(), path=self.save_info_path)

    def train_step(self, train_data: tf.Tensor, target_data: tf.Tensor):
        with tf.GradientTape() as tape:
            prediction = self.model.pipeline(input_data=train_data)
            loss_value = self.loss_function(prediction=prediction, target_data=target_data)
            weights    = self.weights_operator.collect_weights()
            tape.watch(weights)
        gradients = tape.gradient(loss_value, weights)
        self.optimizer.apply_gradients(zip(gradients, weights))
        return loss_value

    @tf.function(experimental_relax_shapes=True)
    def graph_mode(self,
                   train_data: tf.Tensor,
                   target_data: tf.Tensor):
        return self.train_step(train_data=train_data, target_data=target_data)

    def eager_mode(self,
                   train_data: tf.Tensor,
                   target_data: tf.Tensor):
        return self.train_step(train_data=train_data, target_data=target_data)

    def get_weights(self) -> dict:
        return self.weights_operator.weights

    def will_save(self, epoch_number: int) -> bool:
        if self.save_times is None:
            return False
        return (epoch_number + 1) % self.save_times == 0

    @tf.function(experimental_relax_shapes=True)
    def distributed_train_step(self,
                               train_data: tf.Tensor,
                               target_data: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        accuracies, per_replica_losses = self.mirrored_strategy.run(self.train_step, args=(train_data, target_data))
        return (self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, accuracies, axis=None),
                self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None))

    def save(self, save_obj: dict, path: str) -> NoReturn:
        with open(path, 'w') as fp:
            json.dump(save_obj, fp)


