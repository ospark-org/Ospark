from ospark.trainer import *
from ospark.nn.loss_function import LossFunction
from ospark import Model
import tensorflow as tf
import time


class ExdeepTransformerTrainer(Trainer):

    def __init__(self,
                 data_generator: DataGenerator,
                 model: Model,
                 epoch_number: int,
                 optimizer: Optimizer,
                 loss_function: LossFunction,
                 save_delegate: Optional[SaveDelegate]=None,
                 save_times: Optional[int]=None,
                 save_path: Optional[str]=None,
                 use_profiling_phase: Optional[bool]=True,
                 use_auto_graph: Optional[bool]=True,
                 save_init_weights: Optional[bool]=False,
                 init_weights_path: Optional[str]=None
                 ):
        super().__init__(model=model,
                         data_generator=data_generator,
                         epoch_number=epoch_number,
                         optimizer=optimizer,
                         save_delegate=save_delegate,
                         save_path=save_path,
                         save_times=save_times,
                         use_auto_graph=use_auto_graph,
                         loss_function=loss_function)
        self._use_profiling_phase         = use_profiling_phase
        self._save_init_weights           = save_init_weights
        self._init_weights_path           = init_weights_path or self.save_path.split(".")[0] + "_init.json"
        self._log_file                    = open(self.save_path.split(".")[0] + ".txt", 'w')

    @property
    def use_profiling_phase(self) -> bool:
        return self._use_profiling_phase

    @property
    def log_file(self):
        return self._log_file

    @property
    def save_init_weights(self) -> bool:
        return self._save_init_weights

    @property
    def init_weights_path(self) -> str:
        return self._init_weights_path

    @tf.function(input_signature=[
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64)])
    def graph_mode(self,
                   train_data: tf.Tensor,
                   target_data: tf.Tensor):
        return self.train_step(train_data=train_data, target_data=target_data)

    def start(self) -> NoReturn:
        if self.save_init_weights:
            weights = self.weights_operator.weights
            self.save(weights=weights, path=self.init_weights_path)

        if self.use_profiling_phase:
            dataset = next(iter(self.data_generator))
            profiling_encoder_input, profiling_decoder_input = dataset.training_data, dataset.target_data

            print("Profiling phase start.")
            self.model.profiling_phase(encoder_input=profiling_encoder_input,
                                       decoder_input=profiling_decoder_input[:, :-1])
            print("Profiling phase end.")

            print("=" * 24)
        else:
            self.model.back_to_standard()
        print("Training phase start.")
        self.training_process(epoch_number=self.epoch_number)
        print("Training phase end.")
        print("=" * 24)

    def training_process(self, epoch_number) -> NoReturn:
        for epoch in range(epoch_number):
            total_loss_value = 0
            training_count   = 0
            total_accuracies = 0
            start_time       = time.time()
            for batch, dataset in enumerate(self.data_generator):
                training_data, target_data = dataset.training_data, dataset.target_data
                accuracies, loss_value = self.training_method(training_data, target_data)
                total_accuracies += accuracies
                total_loss_value += loss_value
                training_count   += 1
                print(loss_value)
                if (batch + 1) % 50 == 0:
                    print(f"Epoch {epoch + 1} Batch {batch} Loss {loss_value:.4f} Accuracy {accuracies:.4f}")

            print(f'Epoch {epoch + 1} '
                  f'Loss {total_loss_value / training_count:.4f} '
                  f'Accuracy {total_accuracies / training_count:.4f}', file=self.log_file)
            print(f'Epoch {epoch + 1} '
                  f'Loss {total_loss_value / training_count:.4f} '
                  f'Accuracy {total_accuracies / training_count:.4f}')
            print(f'Time taken for 1 epoch: {time.time() - start_time:.2f} secs\n')
            if self.will_save(epoch_number=epoch):
                self.save_delegate.save(weights=self.weights_operator.weights, path=self.save_path)
        self.save(weights=self.weights_operator.weights, path=self.save_path)
        self.log_file.close()

    def calculate_accuracy(self, prediction: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        accuracies = tf.math.equal(target, tf.argmax(prediction, axis=-1))
        mask       = tf.logical_not(tf.math.equal(target, 0))
        accuracies = tf.logical_and(mask, accuracies)
        accuracies = tf.cast(accuracies, tf.float32)
        mask       = tf.cast(mask, tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    def train_step(self, train_data: tf.Tensor, target_data: tf.Tensor):
        with tf.GradientTape() as tape:
            weights    = self.model.training_weights
            prediction = self.model.pipeline(encoder_input=train_data, decoder_input=target_data[:, :-1])
            loss_value = self.loss_function(prediction=prediction, target_data=target_data[:, 1:])
            accuracies = self.calculate_accuracy(prediction=prediction, target=target_data[:, 1:])
            tape.watch(weights)
        gradients  = tape.gradient(loss_value, weights)
        self.optimizer.apply_gradients(zip(gradients, weights))
        return accuracies, loss_value

