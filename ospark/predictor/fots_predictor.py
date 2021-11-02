from ospark.predictor import Predictor
from ospark.detection_model.pixel_wise import PixelWiseDetection
from ospark.recognition_model.text_recognition import TextRecognition
from ospark.utility.roi_rotate import RoIRotate
from ospark.utility.non_max_suppression import NonMaxSuppression
from typing import Tuple, Optional
import tensorflow as tf
import numpy as np


class FOTSPredictor(Predictor):

    def __init__(self,
                 detection_model: PixelWiseDetection,
                 recognition_model: TextRecognition,
                 corpus: dict,
                 roi_image_height: Optional[int]=8,
                 score_threshold: Optional[float]=0.7,
                 iou_threshold: Optional[float]=0.5):
        self._detection_model   = detection_model
        self._recognition_model = recognition_model
        self._corpus            = corpus
        self._score_threshold   = score_threshold
        self._rotator           = RoIRotate(batch_size=1, target_height=roi_image_height)
        self._nms_processor     = NonMaxSuppression(iou_threshold=iou_threshold)

    @property
    def detection_model(self) -> PixelWiseDetection:
        return self._detection_model

    @property
    def recognition_model(self) -> TextRecognition:
        return self._recognition_model

    @property
    def score_threshold(self) -> float:
        return self._score_threshold

    @property
    def corpus(self) -> dict:
        return self._corpus

    @property
    def rotator(self) -> RoIRotate:
        return self._rotator

    @property
    def nms_processor(self) -> NonMaxSuppression:
        return self._nms_processor


    def predict(self, input_data: tf.Tensor):
        image_size = np.shape(input_data.numpy())[:2]
        bbox_prediction, feature_map = self.detection_model(input_data=input_data)

        bboxes_datasets = self.create_bboxes_datasets(bbox_prediction=bbox_prediction)
        nms_result      = self.nms_processor.start(bboxes_datasets=bboxes_datasets, image_size=image_size)
        text_boxes, _   = zip(*map(self.rotator.textbox_from(image=feature_map), nms_result))
        text_prediction = []
        for text_box in text_boxes:
            recognize_prediction = self.recognition_model(input_data=text_box)
            recognize_prediction = tf.transpose(recognize_prediction, [1, 0, 2])
            prediction = tf.nn.ctc_beam_search_decoder(inputs=recognize_prediction,
                                                       sequence_length=[tf.shape(recognize_prediction)[0]])
            word_indices = tf.squeeze(tf.sparse.to_dense(prediction)).numpy()
            text_prediction.append("".join([self.corpus[i] for i in word_indices]))
        return nms_result, text_prediction

    def create_bboesx_datasets(self, bbox_prediction: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        score_map, bbox_map, angle_map = [prediction.numpy() for prediction in bbox_prediction]

        indices           = np.where(score_map > self.score_threshold)
        predict_bbox      = bbox_map[indices[0], indices[1], :]
        predict_angle     = angle_map[indices[0], indices[1], :]
        horizontal_vector = np.concatenate([predict_bbox[:, [0, 2]] * np.sin(predict_angle),
                                            predict_bbox[:, [0, 2]] * np.cos(predict_angle)],
                                           axis=-1)
        left_vectors      = horizontal_vector[:, [0, 2]][:, np.newaxis, :]
        right_vectors     = horizontal_vector[:, [1, 3]][:, np.newaxis, :]
        vertical_vector   = np.concatenate([predict_bbox[:, [1, 3]] * np.cos(predict_angle),
                                            predict_bbox[:, [1, 3]] * np.sin(predict_angle)],
                                            axis=-1)
        top_vectors       = vertical_vector[:, [0, 2]][:, np.newaxis, :]
        bottom_vector     = vertical_vector[:, [1, 3]][:, np.newaxis, :]
        vectors           = np.concatenate([left_vectors, top_vectors, right_vectors, bottom_vector], axis=1)

        bbox_datasets = []
        for index in range(len(indices[0])):
            score = score_map[indices[0][index], indices[1][index], 0]
            reference_coordinate = np.array([indices[0][index], indices[1][index]])
            bbox_points = self.calculate_bbox_points(reference_coordinate=reference_coordinate,
                                                     vectors=vectors[index, :, :])
            bbox_datasets.append((score, bbox_points))
        return bbox_datasets

    def calculate_bbox_points(self, reference_coordinate: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        return reference_coordinate + (vectors[[0, 1, 2, 3], :] + vectors[[1, 2, 3, 0], :])
