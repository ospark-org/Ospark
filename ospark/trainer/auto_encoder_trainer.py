from ospark.trainer import *


class AutoencoderTrainer(Trainer):

    def __init__(self,
                 model: Model,
                 data_generator: DataGenerator,
                 epoch_number: int,
                 optimizer: Optimizer,
                 loss_function: Union[LossFunction, Dict[str, LossFunction]],
                 presentation_of_loss_value: Optional[int]=None,
                 save_delegate: Optional[SaveDelegate]=None,
                 save_times: Optional[int]=None,
                 save_weights_path: Optional[str]=None,
                 use_auto_graph: Optional[bool]=True,
                 use_multi_gpu: Optional[bool]=None,
                 devices: Optional[List[str]]=None,
                 logger: Optional[Logger]=None):
        super().__init__(model=model,
                         data_generator=data_generator,
                         epoch_number=epoch_number,
                         optimizer=optimizer,
                         loss_function=loss_function,
                         presentation_of_loss_value=presentation_of_loss_value,
                         save_weights_path=save_weights_path,
                         save_times=save_times,
                         save_delegate=save_delegate,
                         use_auto_graph=use_auto_graph,
                         use_multi_gpu=use_multi_gpu,
                         devices=devices,
                         logger=logger)

    @tf.function(experimental_relax_shapes=True)
    def graph_mode(self,
                   train_data: tf.Tensor,
                   timestamps: tf.Tensor):
        return self.train_step(train_data=train_data, timestamps=timestamps)

    def eager_mode(self,
                   train_data: tf.Tensor,
                   timestamps: tf.Tensor):
        return self.train_step(train_data=train_data, timestamps=timestamps)

    def training_process(self) -> NoReturn:
        for epoch in range(self.epoch_number):
            total_loss_value = 0
            training_count   = 0
            start_time       = time.time()
            for step, dataset in enumerate(self.data_generator):
                training_data, timestamps = dataset.training_data, dataset.timestamps
                loss_value = self.training_method(training_data, timestamps)
                total_loss_value += loss_value
                training_count   += 1
                if self._presentation_of_loss_value is not None and step % self._presentation_of_loss_value == 0:
                    logging.info(f"step: {step}, loss value : {total_loss_value / training_count}")
                    logging.info("estimated time pre epoch: ", self.data_generator.max_step / (step + 1) * (time.time() - start_time))


            logging.info(f'Epoch {epoch + 1} '
                         f'Loss {total_loss_value / training_count:.4f} ')
            logging.info(f'Time taken for 1 epoch: {time.time() - start_time:.2f} secs\n')
            if self.will_save(epoch_number=epoch) and self.save_weights_path is not None:
                self.save_delegate.save(save_obj=self.weights_operator.weights, path=self.save_weights_path)

        if self.save_weights_path is not None:
            self.save_delegate.save(save_obj=self.weights_operator.weights, path=self.save_weights_path)

    def train_step(self, train_data: tf.Tensor, timestamps: tf.Tensor):
        with tf.GradientTape() as tape:
            prediction, target_data = self.model.pipeline(input_data=train_data, batch_timestamps=timestamps)

            loss_value = self.loss_function(prediction=prediction, target_data=target_data)
            weights    = self.weights_operator.collect_weights()
            tape.watch(weights)
        gradients = tape.gradient(loss_value, weights)
        self.optimizer.apply_gradients(zip(gradients, weights))
        return loss_value