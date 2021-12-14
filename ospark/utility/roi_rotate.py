from typing import List, NoReturn, Tuple, Callable, Union, Optional
import tensorflow as tf
import math
from PIL import Image


class RoIRotate:

    def __init__(self, batch_size: int, target_height: int):
        self._batch_size    = batch_size
        self._target_height = target_height

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def target_height(self) -> tf.Tensor:
        return tf.constant(self._target_height)

    def start(self,
              images: tf.Tensor,
              bbox_points: tf.Tensor,
              target_words: List[List[str]]
              ) -> Tuple[tf.Tensor, int, List[str], List[tf.Tensor]]:
        """
        Start roi rotate.

        Args:
        images: tf.Tensor
            Input is images or feature maps, type is tensor.
        bbox_points: tf.Tensor
            Bounding box points, format is [[x1, y1],[x2, y2],[x3, y3],[x4, y4]].
        target_words: List[List[str]]
            Target word of each text box.

        Returns:
            feature_maps: tf.Tensor
                Result of RoI rotate, feature map is text image.
            textbox_number: int
            words_stacked: List[str]
                Stacked of all target words.
            wrapped_widths: List[tf.Tensor]
                Stacked of all textbox widths.
        """

        words_stacked = []
        datasets      = []
        _ = [
            [datasets.extend(map(self.textbox_from(image=image), bounding_boxes_points)),
            words_stacked.extend(words)]
            for image, bounding_boxes_points, words in zip(images, bbox_points, target_words)
            if bounding_boxes_points != []
            ]

        wrapped_widths = [dataset[1] for dataset in datasets]  # datasets is ((textbox, width), (textbox, width), ...)
        max_width      = tf.reduce_max(tf.convert_to_tensor(wrapped_widths), axis=0)
        textbox_number = len(datasets)
        feature_maps   = tuple(map(self.padding_to(max_width=max_width), datasets))
        feature_maps   = tf.concat(feature_maps, axis=0)
        return feature_maps, textbox_number, words_stacked, wrapped_widths

    def textbox_from(self, image: tf.Tensor) -> Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        """
        Preload image.

        Args:
            image: tf.Tensor

        Returns:
            affine_transformation:  Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]
        """

        image_height, image_width = image.shape[0], image.shape[1]

        def _demarcated(bbox_points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            result = self.crop_image(image=image,
                                     bbox_points=bbox_points,
                                     image_height=image_height,
                                     image_width=image_width)

            if result is not None:
                crop_img, offset_width, offset_height, subimage_width, subimage_height = result

                zoom_rate, textbox_width  = self.calculate_resize_scale(points=bbox_points)
                target_image_coordinate   = self.create_coordinate_matrix(image_height=self.target_height, image_width=textbox_width)
                original_image_coordinate = self.create_coordinate_matrix(image_height=subimage_height, image_width=subimage_width)

                affine_matrix     = self.create_affine_matrix(zoom_rate=zoom_rate,
                                                              bbox_points=bbox_points,
                                                              left_top_point=[bbox_points[0, 1] - offset_height,
                                                                              bbox_points[0, 0] - offset_width])
                # Mapping to original coordinate.
                mapping_coordinate = tf.matmul(target_image_coordinate, tf.transpose(affine_matrix, [1, 0]))
                textbox           = self.roi_rotate(crop_img=crop_img,
                                                    mapping_coordinates=mapping_coordinate,
                                                    original_coordinate=original_image_coordinate,
                                                    textbox_width=textbox_width)
                return textbox, textbox_width
        return _demarcated

    def calculate_resize_scale(self, points: tf.Tensor):
        top_bound_length  = self.calculate_length(point_1=points[1, :], point_2=points[0, :])
        left_bound_length = self.calculate_length(point_1=points[3, :], point_2=points[0, :])

        zoom_rate     = tf.cast(self.target_height, dtype=tf.float32) / left_bound_length
        textbox_width = tf.cast(tf.math.ceil(top_bound_length * zoom_rate), dtype=tf.int32)
        return zoom_rate, textbox_width

    def create_affine_matrix(self,
                             zoom_rate: tf.float32,
                             bbox_points: tf.Tensor,
                             left_top_point: List[int]) -> tf.Tensor:
        angle  = self.calculate_angle(start_point=bbox_points[0, :], end_point=bbox_points[1, :])
        y_point, x_point = -1 * tf.cast(left_top_point, dtype=tf.float32)
        affine_matrix = zoom_rate * tf.convert_to_tensor([[tf.cos(angle), tf.sin(angle), (x_point * tf.cos(angle)) + (y_point * tf.sin(angle))],
                                                   [-tf.sin(angle), tf.cos(angle), (y_point * tf.cos(angle)) - (x_point * tf.sin(angle))],
                                                   [0, 0, 1 / zoom_rate]],
                                                  dtype=tf.float32)
        affine_matrix = tf.linalg.inv(affine_matrix)
        return affine_matrix

    def create_coordinate_matrix(self, image_height: Union[int, tf.Tensor], image_width: Union[int, tf.Tensor]) -> tf.Tensor:
        x_axes = tf.tile(tf.range(1, image_width + 1, dtype=tf.float32)[tf.newaxis, :, tf.newaxis], [image_height, 1, 1])
        y_axes = tf.tile(tf.range(1, image_height + 1, dtype=tf.float32)[:, tf.newaxis, tf.newaxis], [1, image_width, 1])
        z_axes = tf.ones(shape=[image_height, image_width, 1], dtype=tf.float32)
        return tf.concat([x_axes, y_axes, z_axes], axis=-1)

    def calculate_length(self, point_1: tf.Tensor, point_2: tf.Tensor) -> tf.Tensor:
        return tf.linalg.norm(tf.cast(point_1 - point_2, dtype=tf.float32), axis=0)

    def calculate_angle(self,
                        start_point: tf.Tensor,
                        end_point: tf.Tensor,
                        axis: Optional[str]="x") -> tf.Tensor:
        """
        Calculate the angle between the line segment and the x or y axis.

        Args:
            start_point: tf.Tensor
            end_point: tf.Tensor
            axis: Optional[str]="x"
                It is optional x or y, default is x axis.

        Returns:
            angle: tf.Tensor
                Angle of line segment.
        """
        vector = tf.cast(end_point - start_point, dtype=tf.float32)
        length = tf.linalg.norm(vector)

        x_partition, y_partition = vector
        if axis == "x":
            if x_partition != 0:
                angle = tf.acos(x_partition / length) / tf.cond(y_partition >= 0, lambda: 1, lambda: -1)
            else:
                angle = tf.cond(y_partition >= 0, lambda: math.pi / 2, lambda: -math.pi / 2)
        elif axis == "y":
            if y_partition != 0:
                angle = tf.acos(y_partition / length) / tf.cond(x_partition >= 0, lambda: 1, lambda: -1)
            else:
                angle = tf.cond(x_partition >= 0, lambda: math.pi / 2, lambda: -math.pi / 2)
        else:
            raise KeyError(f"Undefined axis {axis}, place use \"x\" or \"y\" axis.")
        return angle

    def crop_image(self,
                   image: tf.Tensor,
                   bbox_points: tf.Tensor,
                   image_height: tf.Tensor,
                   image_width: tf.Tensor
                   ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        max_point = tf.cast(tf.reduce_max(bbox_points, axis=0), tf.int32)
        min_point = tf.cast(tf.reduce_min(bbox_points, axis=0), tf.int32)
        offset_width, offset_height = min_point[0], min_point[1]
        target_width, target_height = tf.minimum(max_point[0], image_width) - min_point[0], tf.minimum(max_point[1], image_height) - min_point[1]
        if target_width > 0 and target_height > 0:
            crop_img = tf.cast(tf.image.crop_to_bounding_box(image=image,
                                                             offset_width=offset_width,
                                                             offset_height=offset_height,
                                                             target_width=target_width,
                                                             target_height=target_height), dtype=tf.float32)
            return crop_img, offset_width, offset_height, target_width, target_height

    def roi_rotate(self,
                   crop_img: tf.Tensor,
                   mapping_coordinates: tf.Tensor,
                   original_coordinate: tf.Tensor,
                   textbox_width: tf.Tensor) -> tf.Tensor:
        value_stacked = []
        for mapping_coordinate in tf.reshape(mapping_coordinates, [-1, 3]):
            scale_rate = tf.math.maximum(0, 1 - tf.abs(original_coordinate - mapping_coordinate))
            x_scale_rate, y_scale_rate, _ = tf.split(scale_rate, 3, axis=-1)
            bilinear_interpolation = tf.reduce_sum(tf.reduce_sum(crop_img * x_scale_rate, axis=1) * y_scale_rate[:, 0], axis=0)
            value_stacked.append(bilinear_interpolation)
        del crop_img
        textbox = tf.reshape(value_stacked, [self.target_height, textbox_width, -1])
        return textbox

    def padding_to(self, max_width: tf.int32) -> Callable[[Tuple[tf.Tensor, tf.Tensor]], tf.Tensor]:
        def zero_padded(dataset: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
            textbox, textbox_width = dataset

            padded_textbox = tf.pad(textbox[tf.newaxis, :, :, :], paddings=[[0, 0],
                                                                            [0, 0],
                                                                            [0, max_width - textbox_width],
                                                                            [0, 0]])
            return padded_textbox
        return zero_padded


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
