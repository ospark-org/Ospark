from __future__ import annotations
from ospark.data.generator import DataGenerator
from ospark.utility.padding_method import PaddingManager
from typing import List, Optional, Set, Tuple
from PIL import Image
import tensorflow as tf
import numpy as np
import random
import copy
import cv2
import os


class FOTSDataGenerator(DataGenerator):

    def __init__(self,
                 training_data_folder: str,
                 target_data_folder: str,
                 training_files_name: List[str],
                 target_files_name: List[str],
                 batch_size: int,
                 filter_height: int,
                 filter_words: Set[str],
                 target_size: list,
                 image_shrunk_scale: float,
                 padding_method: Optional[str]="center",
                 bbox_shrunk_scale: Optional[float]=0.3,
                 initial_step: Optional[int]=None,
                 use_shuffle: Optional[bool]=False):
        """

        :param training_files_name:
        :param target_files_name:
        :param batch_size:
        :param filter_height:
        :param filter_words:
        image_size: np.ndarray
            image size [width, height]
        :param image_shrunk_scale:
        padding_method: str
            Method has center, upper_left, upper_right, lower_left, lower_right. Default use "center".
        :param bbox_shrunk_scale:
        :param initial_step:
        """

        super().__init__(training_data=training_files_name,
                         target_data=target_files_name,
                         batch_size=batch_size,
                         initial_step=initial_step)
        self._training_data_folder  = training_data_folder
        self._target_data_folder    = target_data_folder
        self._filter_height         = filter_height
        self._filter_words          = filter_words
        self._image_shrunk          = image_shrunk_scale
        self._bbox_shrunk_scale     = bbox_shrunk_scale
        self._target_size           = np.array(target_size)
        self._padding               = PaddingManager(padding_method)
        self._image_shrunk_size     = (np.array(target_size).astype(np.float32) * image_shrunk_scale).astype(np.int32)
        self._blank_image           = np.zeros(shape=[self.image_shrunk_size[1], self.image_shrunk_size[0], 1])
        self._coordinate_matrix     = self.create_coordinate_matrix()
        self._basic_move_distance   = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        self._indexed_training_data = self.build_file_index(self.training_data)
        self._indexed_target_data   = self.build_file_index(self.target_data)
        self._data_indices          = [i for i in self.indexed_training_data.keys()]
        self._use_shuffle           = use_shuffle
        if self.use_shuffle:
            random.shuffle(self._data_indices)

    @property
    def filter_height(self) -> int:
        return self._filter_height

    @property
    def filter_words(self) -> Set[str]:
        return self._filter_words

    @property
    def image_shrunk(self) -> float:
        return self._image_shrunk

    @property
    def bbox_shrunk_scale(self) -> float:
        return self._bbox_shrunk_scale

    @property
    def target_size(self) -> np.ndarray:
        return self._target_size

    @property
    def padding(self) -> PaddingManager:
        return self._padding

    @property
    def image_shrunk_size(self) -> np.ndarray:
        return self._image_shrunk_size

    @property
    def blank_image(self) -> np.ndarray:
        return self._blank_image

    @property
    def coordinate_matrix(self) -> np.ndarray:
        return self._coordinate_matrix

    @property
    def training_data_folder(self) -> str:
        return self._training_data_folder

    @property
    def target_data_folder(self) -> str:
        return self._target_data_folder

    @property
    def data_indices(self) -> list:
        return self._data_indices

    @property
    def indexed_training_data(self) -> dict:
        return self._indexed_training_data

    @property
    def indexed_target_data(self) -> dict:
        return self._indexed_target_data

    @property
    def use_shuffle(self) -> bool:
        return self._use_shuffle

    def create_coordinate_matrix(self) -> np.ndarray:
        """
        This function doing create coordinate matrix.

        Returns:
             coordinate_matrix: np.ndarray
                shape is [y_size, x_size, axis_number]
        """
        x_axis = np.tile(np.arange(self.image_shrunk_size[0])[np.newaxis, :, np.newaxis], [self.image_shrunk_size[1], 1, 1])
        y_axis = np.tile(np.arange(self.image_shrunk_size[1])[:, np.newaxis, np.newaxis], [1, self.image_shrunk_size[0], 1])
        return np.concatenate([x_axis, y_axis], axis=-1)

    def __iter__(self) -> FOTSDataGenerator:
        return self

    def __next__(self) -> Tuple[tf.Tensor, tf.Tensor, List[List[str]], List[List[np.ndarray]]]:
        if self.step + 1 < self.max_step:
            training_data, target_image, words, bbox_points = self.get_data()
            self._step += 1
            return training_data, target_image, words, bbox_points
        self.reset()
        raise StopIteration()

    def get_data(self) -> Tuple[tf.Tensor, tf.Tensor, List[List[str]], List[List[np.ndarray]]]:
        start_point = self.batch_size * self.step
        end_point   = min(len(self.training_data), start_point + self.batch_size)
        indices     = self.data_indices[start_point: end_point]

        training_data_paths = [self.indexed_training_data[index] for index in indices]
        training_data       = self.process_training_data(training_data_paths)
        target_data_paths   = [self.indexed_target_data[index] for index in indices]

        target_image, words, bbox_points = self.process_target_data(target_data_paths)
        return training_data, target_image, words, bbox_points

    def process_training_data(self, paths: List[str]) -> tf.Tensor:
        imgs = []
        for path in paths:
            img = Image.open(os.path.join(self.training_data_folder, path))
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
            detection_map = np.zeros(shape=[self.image_shrunk_size[1], self.image_shrunk_size[0], 6])
            fp = open(os.path.join(self.target_data_folder, path), 'r', encoding="utf-8-sig")
            points = []
            words  = []
            for datasets in fp:
                target      = datasets.strip().split(",")
                bbox_points = np.reshape(np.around(np.array([int(j) for j in target[:8]]) * self.image_shrunk).astype(np.int32),
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
                if height / self.image_shrunk < self.filter_height or width / self.image_shrunk < self.filter_height:
                    continue

                detection_map += self.create_detection_map(bbox_points=bbox_points)
                target_word    = ",".join(target[8:])

                if target_word in self.filter_words:
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
        clockwise_vector        = (bbox_points[[1, 2, 3, 0], :] - bbox_points) * self.bbox_shrunk_scale / 2
        counterclockwise_vector = (bbox_points[[3, 0, 1, 2], :] - bbox_points) * self.bbox_shrunk_scale / 2
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

    def build_file_index(self, files: List[str]) -> dict:
        files.sort()
        filtered_file = [string for string in files if any([char.isdigit() for char in string])]
        datasets = map(self._indexed, filtered_file)
        return dict(datasets)

    def _indexed(self, file_name: str) -> Tuple[int, str]:
        number_index = np.where(np.array([char.isdigit() for char in file_name]) == True)[0]
        index        = int("".join([file_name[i] for i in number_index]))
        return index, file_name


if __name__ == "__main__":
    # 檢查結果是否正確，檢查 image size 需不需要前後對調，sorted file name, random sample, croups

    training_data_folder = "/Users/abnertsai/Documents/Ospark/ospark/samples/dataset/ICDAR/training_data"
    target_data_folder   = "/Users/abnertsai/Documents/Ospark/ospark/samples/dataset/ICDAR/ch4_training_localization_transcription_gt"
    training_list        = os.listdir(training_data_folder)
    target_list          = os.listdir(target_data_folder)
    data_generator       = FOTSDataGenerator(training_data_folder=training_data_folder,
                                             target_data_folder=target_data_folder,
                                             training_files_name=training_list,
                                             target_files_name=target_list,
                                             batch_size=4,
                                             filter_height=8,
                                             filter_words={"###"},
                                             target_size=[1280, 720],
                                             image_shrunk_scale=0.25)
    for training_data, target_image, words, bbox_points in data_generator:
        print(words, bbox_points)
        for imag, target_image, bbox_point in zip(training_data, target_image, bbox_points):
            # img = Image.fromarray(imag.numpy().astype(np.uint8))
            # img.show()
            index = np.where(target_image[:, :, 2].numpy() > 0)
            # print(target_image[:, :, 1:5].numpy()[index])
            # tar_img = Image.fromarray(target_image[:, :, 1].numpy().astype(np.uint8) * 20)
            # tar_img = tar_img.resize((1280, 720), Image.BILINEAR)
            # tar_img.show()
        break




