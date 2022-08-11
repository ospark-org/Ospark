from __future__ import annotations
from ospark.data.generator import DataGenerator
from ospark.utility.padding_method import PaddingManager
from typing import List, Optional, Set, Tuple
from ospark.data.path import DataPath
from ospark.data.folder import DataFolder
from PIL import Image
import tensorflow as tf
import numpy as np
import pathlib
import random
import copy
import cv2
import os


class FOTSDataGenerator(DataGenerator):

    class Dataset:

        def __init__(self):
            self._training_data = None
            self._target_data   = None
            self._words         = None
            self._bbox_points   = None

        @property
        def training_data(self) -> tf.Tensor:
            return self._training_data

        @property
        def target_data(self) -> tf.Tensor:
            return self._target_data

        @property
        def words(self) -> List[List[str]]:
            return self._words

        @property
        def bbox_points(self) -> List[List[np.ndarray]]:
            return self._bbox_points

        def data_setting(self,
                         training_data: tf.Tensor,
                         target_data: tf.Tensor,
                         words: List[List[str]],
                         bbox_points: List[List[np.ndarray]]):
            self._training_data = training_data
            self._target_data   = target_data
            self._words         = words
            self._bbox_points   = bbox_points

    def __init__(self,
                 data_folder: DataFolder,
                 batch_size: int,
                 height_threshold: int,
                 ignore_words: Set[str],
                 target_size: list,
                 image_shrink_scale: float,
                 padding_method: Optional[str]="center",
                 bbox_shrink_scale: Optional[float]=0.3,
                 initial_step: Optional[int]=None,
                 use_shuffle: Optional[bool]=False):
        """

        :param training_folder:
        :param labeling_folder:
        :param batch_size:
        :param height_threshold:
        :param ignore_words:
        image_size: np.ndarray
            image size [width, height]
        :param image_shrink_scale:
        padding_method: str
            Method has center, upper_left, upper_right, lower_left, lower_right. Default use "center".
        :param bbox_shrink_scale:
        :param initial_step:
        """

        self._folder                = data_folder
        self._height_threshold      = height_threshold
        self._ignore_words          = ignore_words
        self._shrink_image          = image_shrink_scale
        self._bbox_shrink_scale     = bbox_shrink_scale
        self._target_size           = np.array(target_size)
        self._padding               = PaddingManager(padding_method)
        self._image_shrink_size     = (np.array(target_size).astype(np.float32) * image_shrink_scale).astype(np.int32)
        self._blank_image           = np.zeros(shape=[self.image_shrink_size[1], self.image_shrink_size[0], 1])
        self._coordinate_matrix     = self.create_coordinate_matrix()
        self._basic_move_distance   = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        self._data_indices          = [i for i in self._folder.indexed_training_data.keys()]
        self._data_path             = DataPath(batch_size=batch_size,
                                               total_data=len(self._folder.training_files),
                                               index_table=list(self.folder.indexed_training_data.keys()))
        self._shuffle               = use_shuffle
        if self.shuffle:
            random.shuffle(self._data_indices)

        super().__init__(training_data=self._folder.training_files,
                         target_data=self._folder.labeling_files,
                         batch_size=batch_size,
                         initial_step=initial_step)

    @property
    def folder(self) -> DataFolder:
        return self._folder

    @property
    def height_threshold(self) -> int:
        return self._height_threshold

    @property
    def ignore_words(self) -> Set[str]:
        return self._ignore_words

    @property
    def shrink_image(self) -> float:
        return self._shrink_image

    @property
    def bbox_shrink_scale(self) -> float:
        return self._bbox_shrink_scale

    @property
    def target_size(self) -> np.ndarray:
        return self._target_size

    @property
    def padding(self) -> PaddingManager:
        return self._padding

    @property
    def image_shrink_size(self) -> np.ndarray:
        return self._image_shrink_size

    @property
    def blank_image(self) -> np.ndarray:
        return self._blank_image

    @property
    def coordinate_matrix(self) -> np.ndarray:
        return self._coordinate_matrix

    @property
    def data_indices(self) -> list:
        return self._data_indices

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    def create_coordinate_matrix(self) -> np.ndarray:
        """
        This function doing create coordinate matrix.

        Returns:
             coordinate_matrix: np.ndarray
                shape is [y_size, x_size, axis_number]
        """
        x_axis = np.tile(np.arange(self.image_shrink_size[0])[np.newaxis, :, np.newaxis], [self.image_shrink_size[1], 1, 1])
        y_axis = np.tile(np.arange(self.image_shrink_size[1])[:, np.newaxis, np.newaxis], [1, self.image_shrink_size[0], 1])
        return np.concatenate([x_axis, y_axis], axis=-1)

    def __iter__(self) -> FOTSDataGenerator:
        return self

    def __next__(self) -> Dataset:
        if self.step < self.max_step:
            dataset = self._get_data()
            self._step += 1
            return dataset
        self.reset()
        raise StopIteration()

    def _get_data(self) -> Dataset:
        for index in self._data_path.range(step=self.step):
            training_data_path, labeling_data_path = self.folder.get_files(index=index)
            self._data_path.training_data_paths = training_data_path
            self._data_path.label_data_paths    = labeling_data_path
        training_data = self.process_training_data(self._data_path.training_data_paths)
        target_image, words, bbox_points = self.process_target_data(self._data_path.label_data_paths)
        self.dataset.data_setting(training_data=training_data,
                                  target_data=target_image,
                                  words=words,
                                  bbox_points=bbox_points)
        return self.dataset

    def process_training_data(self, paths: List[str]) -> tf.Tensor:
        imgs = []
        for path in paths:
            img = Image.open(path)
            img = self.padding_and_resize(img)
            imgs.append(img[np.newaxis, :, :, :])
        return tf.cast(tf.concat(imgs, axis=0), dtype=tf.float32)

    def padding_and_resize(self, img: Image.Image) -> np.ndarray:
        input_size = np.array(img.size)
        if all(input_size != self.target_size):
            resize_scale  = (input_size / max(input_size / self.target_size)).astype(np.int32)
            padding_scale = self.target_size - resize_scale
            img = img.resize(size=resize_scale, resample=Image.BILINEAR)
            img = self.padding(image=np.array(img), padding_scale=padding_scale)
        else:
            img = np.array(img)
        return img

    def process_target_data(self, paths: List[str]) -> Tuple[tf.Tensor, List[List[str]], List[List[np.ndarray]]]:
        target_images       = []
        words_stacked       = []
        bbox_points_stacked = []
        for path in paths:
            detection_map = np.zeros(shape=[self.image_shrink_size[1], self.image_shrink_size[0], 6])
            fp = open(path, 'r', encoding="utf-8-sig")
            points = []
            words  = []
            for datasets in fp:
                target      = datasets.strip().split(",")
                bbox_points = np.reshape(np.around(np.array([int(j) for j in target[:8]]) * self.shrink_image).astype(np.int32),
                                         [-1, 2])
                _height      = np.linalg.norm(bbox_points[-1] - bbox_points[0], axis=0)
                _width       = np.linalg.norm(bbox_points[0] - bbox_points[1], axis=0)
                if _height > 2 * _width:
                    height      = _width
                    width       = _height
                    bbox_points = bbox_points[[1, 2, 3, 0], :]
                else:
                    height      = _height
                    width       = _width

                # filtered according origin image size.
                if height / self.shrink_image < self.height_threshold or width / self.shrink_image < self.height_threshold:
                    continue

                detection_map += self.create_detection_map(bbox_points=bbox_points)
                target_word    = ",".join(target[8:])

                if target_word in self.ignore_words:
                    continue
                points.append(bbox_points)
                words.append(target_word)
            target_images.append(detection_map[np.newaxis, :, :, :])
            if points != []:
                words_stacked.append(words)
                bbox_points_stacked.append(points)
        return tf.cast(tf.concat(target_images, axis=0), dtype=tf.float32), words_stacked, bbox_points_stacked

    def create_detection_map(self, bbox_points: np.ndarray) -> np.ndarray:
        shrunk_points   = self.create_shrunk_coordinate(bbox_points)
        score_map       = cv2.fillPoly(copy.copy(self.blank_image), [shrunk_points], 1)
        bbox_map, angle = self.create_bbox_map(score_map=score_map, bbox_points=bbox_points)
        angle_map       = angle * score_map
        return np.concatenate([score_map, bbox_map, angle_map], axis=-1)

    def create_shrunk_coordinate(self, bbox_points: np.ndarray) -> np.ndarray:
        clockwise_vector        = (bbox_points[[1, 2, 3, 0], :] - bbox_points) * self.bbox_shrink_scale / 2
        counterclockwise_vector = (bbox_points[[3, 0, 1, 2], :] - bbox_points) * self.bbox_shrink_scale / 2
        shrunk_vector           = clockwise_vector + counterclockwise_vector
        shrunk_points           = np.round(bbox_points + shrunk_vector).astype(np.int32)
        equal_position          = np.where(shrunk_points == bbox_points)

        shrunk_points[equal_position] += self._basic_move_distance[equal_position]
        return shrunk_points

    def create_bbox_map(self, score_map: np.ndarray, bbox_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        score_map_coordinate = score_map * self.coordinate_matrix
        a, b, c, angle       = self.calculate_coefficient(bbox_points)
        distance = self.calculate_distance(coordinate_matrix=score_map_coordinate, a=a, b=b, c=c)
        # 因為平行四邊形的關係，所以除以 np.cos(angle)
        distance[[1, 3], :, :] = distance[[1, 3], :, :] / np.cos(angle)
        distance = np.transpose(distance, [1, 2, 0]) * score_map
        return distance, angle

    def calculate_coefficient(self, bbox_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        linear equation: ax + by + c = 0
        equation coefficient is a, b and c.
        """

        x     = bbox_points[[1, 2, 3, 0], 0] - bbox_points[[0, 1, 2, 3], 0] + 1e-9
        y     = bbox_points[[1, 2, 3, 0], 1] - bbox_points[[0, 1, 2, 3], 1]
        a     = y / x
        b     = np.array(1)
        c     = bbox_points[[1, 2, 3, 0], 1] - a * bbox_points[[1, 2, 3, 0], 0]
        angle = self.calculate_angle(bbox_points[0], bbox_points[1])
        return -1 * a[:, np.newaxis, np.newaxis], b, -1 * c[:, np.newaxis, np.newaxis], angle

    def calculate_distance(self,
                           coordinate_matrix: np.ndarray,
                           a: np.ndarray,
                           b: np.ndarray,
                           c: np.ndarray) -> np.ndarray:
        """
        linear equation: ax + by + c = 0
        equation coefficient is a, b and c.
        """

        denominator = np.sqrt(np.power(a, 2) + 1)
        numerator   = np.abs(a * coordinate_matrix[:, :, 0] + b * coordinate_matrix[:, :, 1] + c)
        return numerator / denominator

    def calculate_angle(self,
                        start_point: np.ndarray,
                        end_point: np.ndarray) -> np.ndarray:
        vector = end_point - start_point
        length = np.linalg.norm(vector)

        x_partition, y_partition = vector

        coefficient = 1 if y_partition >= 0 else -1
        angle = np.arccos(x_partition / length) / coefficient
        return angle