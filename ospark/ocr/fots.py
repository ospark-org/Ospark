from ospark.trainer import *
from ospark.nn.model import Model
from ospark.utility.roi_rotate import RoIRotate
from typing import List, Tuple, Optional, Union, Any
from ospark.models.pixel_wise import fots_detection_model, PixelWiseDetection
from ospark.nn.loss_function import Dice, Degree, IoU, LossFunction
from ospark.models.text_recognition import TextRecognition, fots_recognition_model
import tensorflow as tf
import numpy as np
import time


class FOTSTrainer(Trainer):

    def __init__(self,
                 data_generator: DataGenerator,
                 detection_model: PixelWiseDetection,
                 recognition_model: TextRecognition,
                 classify_loss_function: LossFunction,
                 bbox_loss_function: LossFunction,
                 degree_loss_function: LossFunction,
                 batch_size: int,
                 epoch_number: int,
                 optimizer: Optimizer,
                 reg_optimizer: Optimizer,
                 corpus: dict,
                 roi_image_height: int=8,
                 recognition_loss_coefficient: Optional[float]=None,
                 save_delegate: Optional[SaveDelegate]=None,
                 save_times: Optional[int]=None,
                 save_path: Optional[str]=None,
                 use_graph: Optional[bool]=True):
        super().__init__(model=detection_model,
                         data_generator=data_generator,
                         epoch_number=epoch_number,
                         optimizer=optimizer,
                         loss_function=None,
                         save_delegate=save_delegate,
                         save_times=save_times,
                         save_path=save_path,
                         use_auto_graph=use_graph,
                         use_multi_gpu=False,
                         devices=None)
        self._roi_rotate                   = RoIRotate(batch_size=batch_size, target_height=roi_image_height)
        self._detection_model              = detection_model
        self._recognition_model            = recognition_model
        self._classify_loss_function       = classify_loss_function
        self._bbox_loss_function           = bbox_loss_function
        self._degree_loss_function         = degree_loss_function
        self._recognition_loss_function    = tf.nn.ctc_loss
        self._corpus                       = corpus
        self._batch_size                   = batch_size
        self._detection_variables          = None
        self._recognition_variables        = None
        self._reg_optimizer                = reg_optimizer
        self._recognition_loss_coefficient = recognition_loss_coefficient or 1.
        self.recognition_model.create()

    @property
    def recognition_model(self) -> Model:
        return self._recognition_model

    @property
    def classify_loss_function(self) -> LossFunction:
        return self._classify_loss_function

    @property
    def bbox_loss_function(self) -> LossFunction:
        return self._bbox_loss_function

    @property
    def degree_loss_function(self) -> LossFunction:
        return self._degree_loss_function

    @property
    def recognition_loss_function(self) -> LossFunction:
        return self._recognition_loss_function

    @property
    def roi_rotate(self) -> RoIRotate:
        return self._roi_rotate

    @property
    def corpus(self) -> dict:
        return self._corpus

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def detection_variables(self) -> List[tf.Tensor]:
        return self._detection_variables

    @property
    def recognition_variables(self) -> List[tf.Tensor]:
        return self._recognition_variables

    @property
    def recognition_loss_coefficient(self) -> float:
        return self._recognition_loss_coefficient

    @property
    def reg_optimizer(self) -> Optimizer:
        return self._reg_optimizer

    @classmethod
    def create_attention_fots(cls,
                              data_generator: DataGenerator,
                              retrained_weights: dict,
                              corpus: dict,
                              batch_size: int,
                              epoch_number: int,
                              optimizer: Optimizer,
                              reg_optimizer: Optimizer,
                              detection_model: Optional[Model]=None,
                              recognition_model: Optional[Model]=None,
                              save_path: Optional[str]=None,
                              save_times: Optional[int]=None,
                              trainable: Optional[bool]=True):
        detection_model   = detection_model or fots_detection_model(trainable=trainable, retrained_weights=retrained_weights)
        recognition_model = recognition_model or fots_recognition_model(class_number=len(corpus),
                                                                        retrained_weights=retrained_weights,
                                                                        scale_rate=4,
                                                                        head_number=8,
                                                                        input_channel=32,
                                                                        sequential_output_channels=[[64, 64],
                                                                                                    [128, 128],
                                                                                                    [256, 256]],
                                                                        trainable=trainable)
        return cls(data_generator=data_generator,
                   detection_model=detection_model,
                   recognition_model=recognition_model,
                   classify_loss_function=Dice(),
                   bbox_loss_function=IoU(),
                   degree_loss_function=Degree(),
                   batch_size=batch_size,
                   epoch_number=epoch_number,
                   optimizer=optimizer,
                   reg_optimizer=reg_optimizer,
                   corpus=corpus,
                   save_path=save_path,
                   save_times=save_times)

    def start(self) -> NoReturn:
        self._detection_variables   = self.weights_operator.collect_weights("detection_model")
        self._recognition_variables = self.weights_operator.collect_weights("recognition_model")
        for i in range(self.epoch_number):
            start = time.time()
            detection_loss_value   = 0
            recognition_loss_value = 0
            for batch, dataset in enumerate(self.data_generator):
                train_data   = dataset.training_data
                target_maps  = dataset.target_data
                target_words = dataset.words
                bbox_points  = dataset.bbox_points

                feature_maps, detection_loss = self.detection_part(input_data=train_data, target_maps=target_maps)

                sub_images, images_number, target_words, total_width = self.roi_rotate.start(images=feature_maps,
                                                                                             bbox_points=bbox_points,
                                                                                             target_words=target_words)

                recognition_loss = self.recognition_part(sub_images=sub_images,
                                                         target_words=target_words,
                                                         images_number=images_number,
                                                         total_width=total_width)

                detection_loss_value += detection_loss[0] + detection_loss[1] + detection_loss[2]
                recognition_loss_value += recognition_loss
            detection_loss_value   /= batch + 1
            recognition_loss_value /= batch + 1
            print(f"detection loss value: {detection_loss_value:.4f}, "
                  f"recognition loss value: {recognition_loss_value:.4f}")
            print("spent time per epoch: ", time.time() - start)

            if self.will_save(epoch_number=i):
                self.save_delegate.save(weights=self.weights_operator.weights)
        self.save_delegate.save(weights=self.weights_operator.weights)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)
    ])
    def detection_part(self,
                       input_data: tf.Tensor,
                       target_maps: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as detection_tape:
            det_predict, feature_maps = self.model.pipeline(input_data)
            loss_value  = self.calculate_detection_loss(prediction=det_predict, target_data=target_maps)
            added_loss  = loss_value[0] + loss_value[1] + loss_value[2]
            gradients   = self.calculate_gradient(tape=detection_tape, loss_value=added_loss, training_variable=self.detection_variables)
        del detection_tape
        self.optimizer.apply_gradients(zip(gradients, self.detection_variables))
        return feature_maps, loss_value

    def calculate_detection_loss(self,
                                 prediction: tf.Tensor,
                                 target_data: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        cls_prediction, bbox_prediction, degree_prediction = prediction

        cls_target, bbox_target, degree_target = self._slice_channel(target_data)

        cls_loss    = self.classify_loss_function(prediction=cls_prediction, target_data=cls_target)
        bbox_loss   = self.bbox_loss_function(prediction=bbox_prediction, target_data=bbox_target, cls_target=cls_target)
        degree_loss = self.degree_loss_function(prediction=degree_prediction, target_data=degree_target, cls_target=cls_target)
        return cls_loss, bbox_loss, degree_loss

    def _slice_channel(self, data) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        cls_channel    = data[:, :, :, 0:1]
        bbox_channel   = data[:, :, :, 1:5]
        degree_channel = data[:, :, :, 5:6]
        return cls_channel, bbox_channel, degree_channel

    def recognition_part(self,
                         sub_images: tf.Tensor,
                         target_words: tf.Tensor,
                         images_number: int,
                         total_width: List[int]) -> Tuple[Union[List[tf.Variable], list], Any]:
        accumulation_gradients        = [tf.Variable(tf.zeros_like(variable.initialized_value()), trainable=False)
                                         for variable in self.recognition_variables]
        total_loss = 0
        counts     = 0
        for images, target_word, images_width in self.sub_images_generator(batch_size=self.batch_size,
                                                                           sub_images=sub_images,
                                                                           target_words=target_words,
                                                                           images_number=images_number,
                                                                           total_width=total_width):

            sparse_target, label_length = self.convert_to_sparse(target_word)
            gradients, loss_value = self.recognition_train_step(training_data=images,
                                                                sparse_target=sparse_target,
                                                                label_length=label_length,
                                                                images_width=images_width)
            total_loss += loss_value
            counts     += 1
            accumulation_gradients = [accumulation_gradients[i].assign_add(var) for i, var in enumerate(gradients)]

        self.reg_optimizer.apply_gradients(zip(accumulation_gradients, self.recognition_variables))
        return total_loss / counts

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.SparseTensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ])
    def recognition_train_step(self,
                               training_data: tf.Tensor,
                               sparse_target: tf.SparseTensor,
                               label_length: tf.int32,
                               images_width: tf.int32):
        with tf.GradientTape() as recognition_tape:
            training_variable = self.weights_operator.collect_weights("recognition_model")
            recognition_tape.watch(training_variable)
            reg_prediction = self.recognition_model.pipeline(input_data=training_data)
            loss_value  = tf.nn.ctc_loss(labels=sparse_target,
                                         logits=reg_prediction,
                                         label_length=label_length,
                                         logit_length=images_width,
                                         blank_index=0,
                                         logits_time_major=False)
            loss_value  = tf.reduce_mean(loss_value)
            loss_value *= self.recognition_loss_coefficient
            gradients   = self.calculate_gradient(tape=recognition_tape,
                                                  loss_value=loss_value,
                                                  training_variable=training_variable)
        del recognition_tape
        return gradients, loss_value


    def calculate_gradient(self,
                           tape: tf.GradientTape,
                           loss_value: tf.Tensor,
                           training_variable: List[tf.Tensor]):
        tape.watch(training_variable)
        gradient = tape.gradient(loss_value, training_variable)

        gradient = [tf.clip_by_value(grad, -1., 1.) for grad in gradient]
        return gradient

    def sub_images_generator(self,
                             batch_size: int,
                             sub_images: tf.Tensor,
                             target_words: List[List[int]],
                             images_number: int,
                             total_width: List[int]) -> tf.Tensor:
        for i in range(0, images_number, batch_size):
            end_point = min(i + batch_size, images_number)
            yield sub_images[i: end_point, :], target_words[i: end_point], total_width[i: end_point]

    def convert_to_sparse(self, target_sequences: List[List[int]], dtype=np.int32):
        """
        Create a sparse matrix of target_sequences.
        Args:
            target_sequences: List[List[int]]
                a list of lists of type dtype where each element is a sequence
        Returns: Tuple[np.ndarray]
            A tuple with (indices, values, shape)
        """

        indices      = []
        values       = []
        label_length = []
        for n, seq in enumerate(target_sequences):
            seq = [self.corpus[char] for char in seq]
            seq_length = len(seq)
            label_length.append(seq_length)
            indices.extend(zip([n] * len(seq), range(seq_length)))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int32)
        values  = np.asarray(values, dtype=dtype)
        shape   = np.asarray([len(target_sequences), indices.max(0)[1] + 1], dtype=np.int32)
        sparse_matrix = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
        return sparse_matrix, label_length
