from __future__ import annotations
from ospark.data.generator import DataGenerator
from ospark.utility.padding_method import ImagePadding
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
                 training_data_path: str,
                 target_data_path: str,
                 training_file_name: List[str],
                 target_file_name: List[str],
                 batch_size: int,
                 filter_height: int,
                 filter_words: Set[str],
                 image_size: list,
                 image_shrunk: float,
                 padding_method: Optional[str]="center",
                 bbox_shrunk_scale: Optional[float]=0.3,
                 initial_step: Optional[int]=None,
                 use_shuffle: bool=False):
        """

        :param training_file_name:
        :param target_file_name:
        :param batch_size:
        :param filter_height:
        :param filter_words:
        image_size: np.ndarray
            image size [width, height]
        :param image_shrunk:
        :param padding_method:
        :param bbox_shrunk_scale:
        :param initial_step:
        """

        super().__init__(training_data=training_file_name,
                         target_data=target_file_name,
                         batch_size=batch_size,
                         initial_step=initial_step)
        self._training_data_path    = training_data_path
        self._target_data_path      = target_data_path
        self._filter_height         = filter_height
        self._filter_words          = filter_words
        self._image_shrunk          = image_shrunk
        self._bbox_shrunk_scale     = bbox_shrunk_scale
        self._image_size            = np.array(image_size)
        self._padding               = ImagePadding(padding_method)
        self._target_image_size     = (np.array(image_size).astype(np.float32) * image_shrunk).astype(np.int32)
        self._origin_image          = np.zeros(shape=[self.target_image_size[1], self.target_image_size[0], 1])
        self._position_matrix       = self.create_position_matrix()
        self._data_indices          = [i for i in range(len(training_file_name))]
        self._indexed_training_data = self.build_file_index(self.training_data)
        self._indexed_target_data   = self.build_file_index(self.target_data)
        self._use_shuffle           = use_shuffle
        if self.use_shuffle:
            random.shuffle(self._data_indices)

    @property
    def filter_height(self) -> int:
        return self._filter_height

    @property
    def filter_words(self) -> List[str]:
        return self._filter_words

    @property
    def image_shrunk(self) -> float:
        return self._image_shrunk

    @property
    def bbox_shrunk_scale(self) -> float:
        return self._bbox_shrunk_scale

    @property
    def image_size(self) -> np.ndarray:
        return self._image_size

    @property
    def padding(self) -> ImagePadding:
        return self._padding

    @property
    def target_image_size(self) -> np.ndarray:
        return self._target_image_size

    @property
    def origin_image(self) -> np.ndarray:
        return self._origin_image

    @property
    def position_matrix(self) -> np.ndarray:
        return self._position_matrix

    @property
    def training_data_path(self) -> str:
        return self._training_data_path

    @property
    def target_data_path(self) -> str:
        return self._target_data_path

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

    def create_position_matrix(self) -> np.ndarray:
        x_axis = np.tile(np.arange(self.target_image_size[0])[np.newaxis, :, np.newaxis], [self.target_image_size[1], 1, 1])
        y_axis = np.tile(np.arange(self.target_image_size[1])[:, np.newaxis, np.newaxis], [1, self.target_image_size[0], 1])
        return np.concatenate([x_axis, y_axis], axis=-1)

    def __iter__(self) -> FOTSDataGenerator:
        return self

    def __next__(self) -> Tuple[tf.Tensor, tf.Tensor, List[List[str]], List[List[np.ndarray]]]:
        if self.step <= self.max_step:
            training_data, target_image, words, bbox_points = self.get_data()
            return training_data, target_image, words, bbox_points
        raise StopIteration()

    def get_data(self) -> Tuple[tf.Tensor, tf.Tensor, List[List[str]], List[List[np.ndarray]]]:
        self._step += 1
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
            img = Image.open(os.path.join(self.training_data_path, path))
            img = self.padding_and_resize(img)
            imgs.append(img[np.newaxis, :, :, :])
        return tf.cast(tf.concat(imgs, axis=0), dtype=tf.float32)

    def padding_and_resize(self, img: Image.Image) -> np.ndarray:
        input_size = np.array(img.size)
        if all(input_size != self.image_size):
            resize_scale  = (input_size / max(input_size / self.image_size)).astype(np.int32)
            padding_scale = self.image_size - resize_scale
            img = img.resize(size=resize_scale, resample=Image.BILINEAR)
            img = self.padding(image=np.array(img), padding_scale=padding_scale)
        else:
            img = np.array(img)
        return img

    def process_target_data(self, paths: List[str]) -> Tuple[tf.Tensor, List[List[str]], List[List[np.ndarray]]]:
        target_imgaes     = []
        total_words       = []
        total_bbox_points = []
        for path in paths:
            detection_map = np.zeros(shape=[self.target_image_size[1], self.target_image_size[0], 6])
            fp = open(os.path.join(self.target_data_path, path), 'r', encoding="utf-8-sig")
            points = []
            words  = []
            for i in fp:
                target = i.strip().split(",")
                bbox_points = np.reshape(np.around(np.array([int(j) for j in target[:8]]) * self.image_shrunk).astype(np.int32), [-1, 2])
                height = np.linalg.norm(bbox_points[-1] - bbox_points[0], axis=0)
                detection_map += self.create_detection_map(bbox_points=bbox_points)
                target_word = ",".join(target[8:])
                if target_word in self.filter_words or height < self.filter_height:
                    continue
                points.append(bbox_points)
                words.append(target_word)
            target_imgaes.append(detection_map[np.newaxis, :, :, :])
            total_words.append(words)
            total_bbox_points.append(points)
        return tf.cast(tf.concat(target_imgaes, axis=0), dtype=tf.float32), total_words, total_bbox_points

    def create_shrunk_points(self, bbox_points: np.ndarray) -> np.ndarray:
        clockwise_vector        = (bbox_points[[1, 2, 3, 0], :] - bbox_points) * self.bbox_shrunk_scale / 2
        counterclockwise_vector = (bbox_points[[3, 0, 1, 2], :] - bbox_points) * self.bbox_shrunk_scale / 2
        shrunk_vector           = clockwise_vector + counterclockwise_vector
        new_points              = np.round(bbox_points + shrunk_vector).astype(np.int32)
        return new_points

    def create_detection_map(self, bbox_points: np.ndarray) -> np.ndarray:
        shrunk_points   = self.create_shrunk_points(bbox_points)
        score_map       = cv2.fillPoly(copy.copy(self.origin_image), [shrunk_points], 1)
        bbox_map, angle = self.create_bbox_map(score_map=score_map, bbox_points=bbox_points)
        angle_map       = angle * score_map
        return np.concatenate([score_map, bbox_map, angle_map], axis=-1)

    def create_bbox_map(self, score_map: np.ndarray, bbox_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        score_map_position = score_map * self.position_matrix
        a, b, c, angle     = self.calculate_coefficient(bbox_points)
        distance = self.calculate_distance(position_matrix=score_map_position,
                                           coefficient_a=a,
                                           coefficient_b=b,
                                           coefficient_c=c)
        distance[[1, 3], :, :] = distance[[1, 3], :, :] / np.cos(angle)
        distance = np.transpose(distance, [1, 2, 0]) * score_map
        return distance, angle

    def calculate_coefficient(self, bbox_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        linear function: y = ax + b
        """
        x     = bbox_points[[1, 2, 3, 0], 0] - bbox_points[[0, 1, 2, 3], 0] + 1e-9
        y     = bbox_points[[1, 2, 3, 0], 1] - bbox_points[[0, 1, 2, 3], 1]
        a     = y / x
        b     = np.array(1)
        c     = bbox_points[[1, 2, 3, 0], 1] - a * bbox_points[[1, 2, 3, 0], 0]
        angle = self.calculate_angle(bbox_points[0], bbox_points[1])
        return -1 * a[:, np.newaxis, np.newaxis], b, -1 * c[:, np.newaxis, np.newaxis], angle

    def calculate_distance(self,
                           position_matrix: np.ndarray,
                           coefficient_a: np.ndarray,
                           coefficient_b: np.ndarray,
                           coefficient_c: np.ndarray) -> np.ndarray:
        denominator = np.sqrt(np.power(coefficient_a, 2) + 1)
        numerator   = np.abs(coefficient_a * position_matrix[:, :, 0] + coefficient_b * position_matrix[:, :, 1] + coefficient_c)
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
        return {int("".join([file[i]
                             for i
                             in np.where(np.array([char.isdigit()
                                                   for char
                                                   in file]) == True)[0]])): file
                for file in filtered_file}

if __name__ == "__main__":
    # 檢查結果是否正確，檢查 image size 需不需要前後對調，sorted file name, random sample, croups
    training_data_folder = "/Users/abnertsai/Documents/ICDAR/ch4/ch4_training_images"
    target_data_folder   = "/Users/abnertsai/Documents/ICDAR/ch4/ch4_training_localization_transcription_gt"
    training_list = os.listdir(training_data_folder)
    target_list = os.listdir(target_data_folder)
    data_generator = FOTSDataGenerator(training_data_path=training_data_folder,
                                       target_data_path=target_data_folder,
                                       training_file_name=training_list,
                                       target_file_name=target_list,
                                       batch_size=4,
                                       filter_height=2,
                                       filter_words="###",
                                       image_size=[1280, 720],
                                       image_shrunk=0.25)
    for training_data, target_image, words, bbox_points in data_generator:
        print(np.shape(training_data), np.shape(target_image))
        print(words, bbox_points)
        # for imag, target_image in zip(training_data, target_image):
            # img = Image.fromarray(imag)
            # img.show()
            # tar_img = Image.fromarray(target_image[:,:,2] * 10)
            # tar_img.show()
        break




