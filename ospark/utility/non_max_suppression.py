from typing import Tuple, List, Optional
from scipy import sparse
import numpy as np
import tensorflow as tf
import copy
import cv2


class NonMaxSuppression:

    def __init__(self,
                 iou_threshold: Optional[float]=0.5):
        self._iou_threshold = iou_threshold

    @property
    def iou_threshold(self) -> float:
        return self._iou_threshold

    def start(self, bboxes_datasets: List[Tuple[tf.Tensor, tf.Tensor]], image_size: List[int]):
        block_image = np.zeros(shape=image_size, dtype=np.uint8)[:, :, np.newaxis]

        datasets = [
            (score,
             sparse.coo_matrix(np.squeeze(cv2.fillPoly(copy.copy(block_image), [bbox_coordinate], 1))),
             bbox_coordinate) for score, bbox_coordinate in bboxes_datasets
        ]
        datasets = sorted(datasets, key=lambda x:-x[0])
        surviving_bbox = []

        while datasets != []:
            observed_score, observed_polygon, bbox_coordinate = datasets.pop(0)
            datasets_len = len(datasets)

            for i in range(datasets_len):
                reference_score, reference_polygon, reference_bbox_coordinate = datasets.pop(0)
                iou = self.calculate_iou(observed_polygon=observed_polygon, reference_polygon=reference_polygon)
                if iou < self.iou_threshold:
                    datasets.append((reference_score, reference_polygon, reference_bbox_coordinate))
            surviving_bbox.append(bbox_coordinate)
        return surviving_bbox

    def calculate_iou(self,  observed_polygon: sparse.coo_matrix, reference_polygon: sparse.coo_matrix):
        added_polygon = observed_polygon + reference_polygon
        intersection   = len((added_polygon == 2).indices)
        union          = len(added_polygon.nonzero()[0])
        iou = intersection / (union + 1e-5)
        return iou