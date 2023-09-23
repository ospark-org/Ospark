from ospark.trainer import *
from ospark.algorithm.blokcwise_making import BlockwiseMasking
from ospark.models.BEiT import BEiT


class BEiTTrainer(Trainer):

    def __init__(self,
                 model: BEiT,
                 tokenizer: Model,
                 data_generator: DataGenerator,
                 epoch_number: int,
                 optimizer: Optimizer,
                 save_weights_path: str,
                 save_info_path: str,
                 loss_function: Union[LossFunction, Dict[str, LossFunction]],
                 masking_ratio: Optional[float]=None,
                 masking_method: Optional[BlockwiseMasking]=None,
                 presentation_of_loss_value: Optional[int]=None,
                 save_delegate: Optional[SaveDelegate]=None,
                 save_times: Optional[int]=None,
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
                         save_info_path=save_info_path,
                         save_delegate=save_delegate,
                         save_times=save_times,
                         use_auto_graph=use_auto_graph,
                         use_multi_gpu=use_multi_gpu,
                         devices=devices,
                         logger=logger)
        self._tokenizer   = tokenizer

        image_size    = model._image_size
        patch_size    = model._patch_size
        masking_ratio = masking_ratio or 0.4
        self._masking_method = masking_method or BlockwiseMasking(patch_matrix_shape=[int(image_size[0] / patch_size[0]),
                                                                                      int(image_size[1] / patch_size[1])],
                                                                  masking_ratio=masking_ratio)

    @property
    def tokenizer(self) -> Model:
        return self._tokenizer

    @property
    def masking_method(self) -> BlockwiseMasking:
        return self._masking_method

    @property
    def model(self) -> BEiT:
        return self._model

    def training_process(self) -> NoReturn:
        for epoch in range(self.epoch_number):
            total_loss_value = []
            start_time       = time.time()
            for step, dataset in enumerate(self.data_generator):
                training_data             = dataset.training_data
                mask_matrix, mask_indices = self.masking_method.pipeline(input_data=training_data)

                tokens         = tf.cast(tf.argmax(self.tokenizer.pipeline(input_data=training_data), axis=-1), tf.float32)
                target         = tokens * tf.cast(tf.math.equal(mask_matrix, 0), dtype=tf.float32)
                loss_value     = self.training_method(training_data, target, mask_matrix[..., tf.newaxis])
                total_loss_value.append(loss_value)
                if self._presentation_of_loss_value is not None and (step + 1) % self._presentation_of_loss_value == 0:
                    self._logger.info(f"step: {step}, loss value : {sum(total_loss_value) / len(total_loss_value):.4f}")
                    self._logger.info("estimated time pre epoch: ", self.data_generator.max_step / (step + 1) * (time.time() - start_time))


            self._logger.info(f'Epoch {epoch + 1} '
                  f'Loss {sum(total_loss_value) / len(total_loss_value):.4f} ')
            self._logger.info(f'Time taken for 1 epoch: {time.time() - start_time:.2f} secs\n')
            if self.will_save(epoch_number=epoch) and self.save_weights_path is not None:
                self.save_delegate.save(save_obj=self.weights_operator.weights, path=self.save_weights_path)

        if self.save_weights_path is not None:
            self.save_delegate.save(save_obj=self.weights_operator.weights, path=self.save_weights_path)

    def train_step(self, train_data: tf.Tensor, target_data: tf.Tensor, mask_matrices: tf.Tensor):
        with tf.GradientTape() as tape:
            prediction = self.model.pipeline(input_data=train_data, mask_matrices=mask_matrices)
            loss_value = self.loss_function(prediction=prediction, target_data=target_data)
            weights    = self.weights_operator.collect_weights()
            tape.watch(weights)
        gradients = tape.gradient(loss_value, weights)
        self.optimizer.apply_gradients(zip(gradients, weights))
        return loss_value


    @tf.function(experimental_relax_shapes=True)
    def graph_mode(self,
                   train_data: tf.Tensor,
                   target_data: tf.Tensor,
                   mask_matrices: tf.Tensor):
        return self.train_step(train_data=train_data, target_data=target_data, mask_matrices=mask_matrices)

    def eager_mode(self,
                   train_data: tf.Tensor,
                   target_data: tf.Tensor,
                   mask_matrices: tf.Tensor):
        return self.train_step(train_data=train_data, target_data=target_data, mask_matrices=mask_matrices)





