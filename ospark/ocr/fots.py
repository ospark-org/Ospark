from ospark.utility.trainer import *
from ospark.nn.model import Model
from ospark.utility.roi_rotate import RoIRotate
from typing import List, Tuple, Optional
from ospark.detection_model.pixel_wise import fots_detection_model, PixelWiseDetection
from ospark.nn.loss_function import Dice, Degree, IoU
from ospark.recognition_model.text_recognition import TextRecognition, fots_recognition_model
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
                 learning_rate: Union[float, LearningRateSchedule],
                 corpus: dict,
                 roi_image_height: int=8):
        super().__init__(data_generator=data_generator,
                         batch_size=batch_size,
                         epoch_number=epoch_number,
                         optimizer=optimizer,
                         learning_rate=learning_rate)
        self._roi_rotate                = RoIRotate(batch_size=batch_size, target_height=roi_image_height)
        self._detection_model           = detection_model
        self._recognition_model         = recognition_model
        self._classify_loss_function    = classify_loss_function
        self._bbox_loss_function        = bbox_loss_function
        self._degree_loss_function      = degree_loss_function
        self._recognition_loss_function = tf.nn.ctc_loss
        self._corpus                    = corpus

    @property
    def detection_model(self) -> Model:
        return self._detection_model

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

    @classmethod
    def create_attention_fots(cls,
                              data_generator: DataGenerator,
                              corpus: dict,
                              batch_size: int,
                              epoch_number: int,
                              optimizer: Optimizer,
                              learning_rate: Union[float, LearningRateSchedule],
                              trainable: Optional[bool]=True):
        detection_model   = fots_detection_model(trainable=trainable)
        recognition_model = fots_recognition_model(class_number=len(corpus),
                                                   scale_rate=4,
                                                   head_number=8,
                                                   input_channel=32,
                                                   sequential_output_channels=[[64, 64], [128, 128], [256, 256]],
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
                   learning_rate=learning_rate,
                   corpus=corpus)

    def start(self):
        self.detection_model.create()
        self.recognition_model.create()
        for i in range(self.epoch_number):
            for train_data, target_maps, target_words, bbox_points in self.data_generator:
                start = time.time()
                detection_training_variable      = self.weights_operator.collect("detection_model")
                recognition_training_variable    = self.weights_operator.collect("recognition_model")
                detection_gradient, feature_maps = self.detection_part(input_data=train_data,
                                                                       training_variable=detection_training_variable,
                                                                       target_maps=target_maps)
                sub_images, images_number, target_words, total_width = self.roi_rotate.start(images=feature_maps,
                                                                                             bbox_points=bbox_points,
                                                                                             target_words=target_words)

                recognition_gradient = self.recognition_part(sub_images=sub_images,
                                                             target_words=target_words,
                                                             images_number=images_number,
                                                             total_width=total_width,
                                                             training_variable=recognition_training_variable)

                self.optimizer.apply_gradients(zip(detection_gradient + recognition_gradient,
                                                   detection_training_variable + recognition_training_variable))
                print("spent time: ", time.time() - start)

    def detection_part(self,
                       input_data: tf.Tensor,
                       training_variable: List[tf.Tensor],
                       target_maps: List[List[int]]) -> Tuple[List[tf.Tensor], tf.Tensor]:
        with tf.GradientTape() as detection_tape:
            det_predict, feature_maps = self.detection_model(input_data)
            loss_value  = self.calculate_detection_loss(prediction=det_predict, target_data=target_maps)
            gradient    = self.calculate_gradient(tape=detection_tape, loss_value=loss_value, training_variable=training_variable)
        del detection_tape
        return gradient, feature_maps

    def calculate_detection_loss(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        cls_prediction, bbox_prediction, degree_prediction = prediction
        cls_target, bbox_target, degree_target             = self._slice_channel(target_data)
        cls_loss    = self.classify_loss_function(prediction=cls_prediction, target_data=cls_target)
        bbox_loss   = self.bbox_loss_function(prediction=bbox_prediction, target_data=bbox_target, cls_target=cls_target)
        degree_loss = self.degree_loss_function(prediction=degree_prediction, target_data=degree_target, cls_target=cls_target)
        print("split detection loss: ", cls_loss, bbox_loss, degree_loss)
        loss_value  = cls_loss + bbox_loss + degree_loss
        return loss_value

    def _slice_channel(self, data) -> Tuple[tf.Tensor]:
        cls_channel    = data[:, :, :, 0:1]
        bbox_channel   = data[:, :, :, 1:5]
        degree_channel = data[:, :, :, 5:6]
        return cls_channel, bbox_channel, degree_channel

    def recognition_part(self,
                         sub_images: tf.Tensor,
                         target_words: tf.Tensor,
                         images_number: int,
                         total_width: List[int],
                         training_variable: List[tf.Tensor]) -> List[tf.Tensor]:
        accumulation_gradients        = [tf.Variable(tf.zeros_like(variable.initialized_value()), trainable=False)
                                         for variable in training_variable]
        total_loss = 0
        for images, target_word, images_width in self.sub_images_generator(batch_size=self.batch_size,
                                                                           sub_images=sub_images,
                                                                           target_words=target_words,
                                                                           images_number=images_number,
                                                                           total_width=total_width):
            with tf.GradientTape() as recognition_tape:
                reg_prediction              = self.recognition_model(input_data=images)
                saprse_target, label_length = self.convert_to_sparse(target_word)
                loss_value                  = tf.reduce_mean(tf.nn.ctc_loss(labels=saprse_target,
                                                                            logits=reg_prediction,
                                                                            label_length=label_length,
                                                                            logit_length=images_width,
                                                                            blank_index=-1,
                                                                            logits_time_major=False))
                total_loss += loss_value
                gradient                    = self.calculate_gradient(tape=recognition_tape,
                                                                      loss_value=loss_value,
                                                                      training_variable=training_variable)
                accumulation_gradients = [accumulation_gradients[i].assign_add(var) for i, var in enumerate(gradient)]
            del recognition_tape
        print("recognition loss: ", total_loss / len(images_width))
        return accumulation_gradients

    def calculate_gradient(self, tape: tf.GradientTape,
                           loss_value: tf.Tensor,
                           training_variable: List[tf.Tensor]):
        tape.watch(training_variable)
        gradient = tape.gradient(loss_value, training_variable)
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
            indices.extend(zip([n] * len(seq), range(1, 2 * seq_length + 1, 2)))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int32)
        values  = np.asarray(values, dtype=dtype)
        shape   = np.asarray([len(target_sequences), indices.max(0)[1] + 2], dtype=np.int32)
        saprse_matrix = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
        return saprse_matrix, label_length
