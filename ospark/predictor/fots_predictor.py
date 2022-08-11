from ospark.predictor import Predictor
from ospark.models.pixel_wise import PixelWiseDetection
from ospark.models.text_recognition import TextRecognition
from ospark.utility.roi_rotate import RoIRotate
from ospark.nn.component.weight import WeightOperator
from ospark.utility.non_max_suppression import NonMaxSuppression
from typing import Tuple, Optional, NoReturn
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
        if len(nms_result) == 0:
            raise ValueError(f"No prediction exceeds the threshold {self.score_threshold}")
        text_boxes, _   = zip(*map(self.rotator.textbox_from(image=tf.squeeze(feature_map)), nms_result))
        text_prediction = []
        for text_box in text_boxes:
            recognize_prediction = self.recognition_model(input_data=text_box[tf.newaxis, ...])
            recognize_prediction = tf.transpose(recognize_prediction, [1, 0, 2])
            prediction, _ = tf.nn.ctc_beam_search_decoder(inputs=recognize_prediction,
                                                       sequence_length=[tf.shape(recognize_prediction)[0]])
            word_indices = tf.sparse.to_dense(prediction[0]).numpy()[0]
            text_prediction.append("".join([self.corpus[i] for i in word_indices]))
        return nms_result, text_prediction

    def create_bboxes_datasets(self, bbox_prediction: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        score_map, bbox_map, angle_map = [prediction.numpy() for prediction in bbox_prediction]

        indices           = np.where(score_map > self.score_threshold)
        predict_bbox      = np.around(bbox_map[0, indices[1], indices[2], :]).astype(np.int32)
        vectors           = np.zeros(shape=[predict_bbox.shape[0], 4, 2])

        vectors[:, [0, 2], 1] = predict_bbox[:, [0, 2]] * [-1, 1]
        vectors[:, [1, 3], 0] = predict_bbox[:, [1, 3]] * [1, -1]
        predict_angle         = np.squeeze(angle_map[0, indices[1], indices[2], :])
        rotation_matrix       = np.concatenate(
                                [np.concatenate([np.cos(predict_angle)[..., np.newaxis, np.newaxis],
                                                 -np.sin(predict_angle)[..., np.newaxis, np.newaxis]], axis=-1),
                                 np.concatenate([np.sin(predict_angle)[..., np.newaxis, np.newaxis],
                                                 np.cos(predict_angle)[..., np.newaxis, np.newaxis]], axis=-1)],
                                axis=-2)
        rotate_vectors        = np.matmul(vectors, rotation_matrix)

        bbox_datasets = []
        for index in range(len(indices[0])):
            score = score_map[indices[0][index], indices[1][index], indices[2][index], indices[3][index]]
            reference_coordinate = np.array([indices[2][index], indices[1][index]])
            bbox_points = self.calculate_bbox_points(reference_coordinate=reference_coordinate, vectors=rotate_vectors[index, ...])
            bbox_datasets.append((score, np.ceil(bbox_points).astype(np.int32)))
        return bbox_datasets

    def calculate_bbox_points(self, reference_coordinate: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        bbox_points = reference_coordinate + (vectors + vectors[[3, 0, 1, 2], :])
        return np.maximum(bbox_points, 0)

    def restore_weights(self, weights: dict) -> NoReturn:
        WeightOperator.restore(weights=weights)
        self.detection_model.create()
        self.recognition_model.create()
