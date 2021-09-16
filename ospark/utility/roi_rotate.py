from typing import List, NoReturn, Tuple, Callable
import tensorflow as tf
from PIL import Image
import copy


class RoIRotate:

    def __init__(self, batch_size: int, target_height: int):
        self._batch_size    = batch_size
        self._target_height = target_height

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def target_height(self) -> int:
        return self._target_height

    def start(self,
              images: tf.Tensor,
              bbox_points: tf.Tensor,
              target_words: List[List[str]]
              ) -> Tuple[tf.Tensor, int, List[str], List[int]]:
        total_words  = []
        total_width  = []
        total_images = []
        for image, points, words in zip(images, bbox_points, target_words):
            if points != []:
                sub_images, width_size = list(zip(*map(self.process_image(image=image), points)))
                total_words  += words
                total_width  += width_size
                total_images += sub_images

        images_number = len(total_images)
        max_width     = tf.reduce_max(tf.convert_to_tensor(total_width), axis=0)
        output_images = list(map(lambda para: self.padding_image(max_width=max_width)(*para),
                                 zip(total_images, total_width)))
        return tf.concat(output_images, axis=0), images_number, total_words, total_width

    def process_image(self, image: tf.Tensor) -> Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        def extract_image(points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            crop_img, offset_width, offset_height, image_width, image_height = self.crop_image(image=image, points=points)

            zoom_rate, width_size  = self.calculate_resize_scale(points=points)
            target_position_matrix = self.create_position_matrix(height=self.target_height, width=width_size)
            image_position_matrix  = self.create_position_matrix(height=image_height, width=image_width)

            affine_matrix     = self.create_affine_matrix(zoom_rate=zoom_rate,
                                                          points=points,
                                                          left_top_point=[points[0, 1] - offset_height,
                                                                          points[0, 0] - offset_width])
            original_position = tf.matmul(target_position_matrix, tf.transpose(affine_matrix, [1, 0]))
            sub_image         = self.roi_rotate(crop_img=crop_img,
                                                original_position=original_position,
                                                image_position_matrix=image_position_matrix,
                                                width_size=width_size)
            return sub_image, width_size
        return extract_image

    def calculate_resize_scale(self, points: tf.Tensor):
        top_bound_length  = self.calculate_length(point_1=points[1, :], point_2=points[0, :])
        left_bound_length = self.calculate_length(point_1=points[3, :], point_2=points[0, :])

        zoom_rate  = self.target_height / left_bound_length
        width_size = tf.cast(tf.math.ceil(top_bound_length * zoom_rate), dtype=tf.int32)
        return zoom_rate, width_size

    def create_affine_matrix(self,
                             zoom_rate: tf.float32,
                             points: tf.Tensor,
                             left_top_point: List[int]) -> tf.Tensor:
        angle = self.calculate_angle(start_point=points[0, :], end_point=points[1, :])
        ty, tx = -1 * tf.cast(left_top_point, dtype=tf.float32)
        matrix = zoom_rate * tf.convert_to_tensor([[tf.cos(angle), tf.sin(angle), (tx * tf.cos(angle)) + (ty * tf.sin(angle))],
                                                   [-tf.sin(angle), tf.cos(angle), (ty * tf.cos(angle)) - (tx * tf.sin(angle))],
                                                   [0, 0, 1 / zoom_rate]],
                                                  dtype=tf.float32)
        inverse_matrix = tf.linalg.inv(matrix)
        return inverse_matrix

    def create_position_matrix(self, height: int, width: int) -> tf.Tensor:
        x_axes = tf.tile(tf.range(1, width + 1, dtype=tf.float32)[tf.newaxis, :, tf.newaxis], [height, 1, 1])
        y_axes = tf.tile(tf.range(1, height + 1, dtype=tf.float32)[:, tf.newaxis, tf.newaxis], [1, width, 1])
        z_axes = tf.ones(shape=[height, width, 1], dtype=tf.float32)
        return tf.concat([x_axes, y_axes, z_axes], axis=-1)

    def calculate_length(self, point_1: tf.Tensor, point_2: tf.Tensor) -> tf.Tensor:
        return tf.linalg.norm(tf.cast(point_1 - point_2, dtype=tf.float32), axis=0)

    def calculate_angle(self,
                        start_point: tf.Tensor,
                        end_point: tf.Tensor,
                        axis: str="x") -> tf.Tensor:
        vector = tf.cast(end_point - start_point, dtype=tf.float32)
        length = tf.linalg.norm(vector)

        x_partition, y_partition = vector
        angle = tf.Tensor
        if axis == "x":
            angle = tf.acos(x_partition / length) / tf.cond(y_partition >= 0, lambda: 1, lambda: -1)
        elif axis == "y":
            angle = tf.acos(y_partition / length) / tf.cond(x_partition >= 0, lambda: 1, lambda: -1)
        return angle

    def crop_image(self,
                   image: tf.Tensor,
                   points: tf.Tensor
                   ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        max_point = tf.cast(tf.reduce_max(points, axis=0), tf.int32)
        min_point = tf.cast(tf.reduce_min(points, axis=0), tf.int32)
        offset_width, offset_height = min_point[0], min_point[1]
        target_width, target_height = max_point[0] - min_point[0], max_point[1] - min_point[1]
        crop_img = tf.cast(tf.image.crop_to_bounding_box(image=image,
                                                         offset_width=offset_width,
                                                         offset_height=offset_height,
                                                         target_width=target_width,
                                                         target_height=target_height), dtype=tf.float32)
        return crop_img, offset_width, offset_height, target_width, target_height

    def roi_rotate(self,
                   crop_img: tf.Tensor,
                   original_position: tf.Tensor,
                   image_position_matrix: tf.Tensor,
                   width_size: tf.Tensor) -> tf.Tensor:
        total_value = []
        for position in tf.reshape(original_position, [-1, 3]):
            scale_rate = tf.math.maximum(0, 1 - tf.abs(image_position_matrix - position))
            x_scale_rate, y_scale_rate, _ = tf.split(scale_rate, 3, axis=-1)
            bi_linear_value = tf.reduce_sum(tf.reduce_sum(crop_img * x_scale_rate, axis=1) * y_scale_rate[:, 0], axis=0)
            total_value.append(bi_linear_value)
        del crop_img
        sub_image = tf.reshape(total_value, [self.target_height, width_size, -1])
        return sub_image

    def padding_image(self, max_width: tf.int32) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        def pad(image: tf.Tensor, image_width: tf.Tensor) -> tf.Tensor:
            img = tf.pad(image[tf.newaxis, :, :, :], paddings=[[0, 0],
                                                               [0, 0],
                                                               [0, max_width - image_width],
                                                               [0, 0]])
            return img
        return pad


if __name__ == "__main__":
    import numpy as np
    file_path = "/Users/abnertsai/Documents/ICDAR/ch4/training_data/img_781.jpg"
    target_path = "/Users/abnertsai/Documents/ICDAR/ch4/ch4_training_localization_transcription_gt/gt_img_781.txt"
    image = np.array(Image.open(file_path))[np.newaxis, :, :, :]
    img = Image.fromarray(image[0])
    # img.show()
    print(image.shape)
    fp = open(target_path, 'r', encoding="utf-8-sig")
    points = []
    words  = []
    for i in fp:
        points.append([int(j) for j in i.strip().split(",")[:8]])
        words.append(i.strip().split(",")[8:])
    rotate = RoIRotate(1, 16)
    images, images_number, total_words, total_width = rotate.start(images=image,
                                                                   bbox_points=[np.array(points).reshape([-1, 4, 2])],
                                                                   target_words=[words])

    for img in images:
        img = Image.fromarray(np.array(img).astype(np.uint8))
        img.show()
