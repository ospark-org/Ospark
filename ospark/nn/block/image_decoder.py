from typing import Optional
from ospark.nn.layers.dense_layer import DenseLayer
from ospark.nn.block import Block, List, NoReturn
from ospark.nn.layers.convolution_layer import ConvolutionLayer
from functools import reduce
import tensorflow as tf


class ImageDecoderBlock(Block):

    def __init__(self,
                 obj_name: str,
                 input_dimension: int,
                 output_dimensions: List[int],
                 is_training: Optional[bool]=None,
                 training_phase: Optional[bool]=None):
        super().__init__(obj_name=obj_name, is_training=is_training, training_phase=training_phase)
        self._input_dimension   = input_dimension
        self._output_dimensions = output_dimensions

    @property
    def input_dimension(self) -> int:
        return self._input_dimension

    @property
    def output_dimensions(self) -> List[int]:
        return self._output_dimensions

    @property
    def use_upsampling(self) -> bool:
        return self._use_upsampling

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError


class DenseLayerDecoder(ImageDecoderBlock):

    def __init__(self,
                 obj_name: str,
                 input_dimension: int,
                 output_dimensions: List[int],
                 patch_number: List[int],
                 patch_size: List[int],
                 is_training: Optional[bool],
                 training_phase: Optional[bool]):
        super().__init__(obj_name=obj_name,
                         input_dimension=input_dimension,
                         output_dimensions=output_dimensions,
                         is_training=is_training,
                         training_phase=training_phase)
        self._dense_layer = DenseLayer(obj_name="decoder_dense",
                                       input_dimension=input_dimension,
                                       hidden_dimension=output_dimensions,
                                       is_training=is_training)
        self._patch_number = patch_number
        self._patch_size   = patch_size

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        batch_size = input_data.shape[0]

        decoder_output = self._dense_layer.pipeline(input_data=input_data)

        image = tf.reshape(decoder_output, shape=[batch_size,
                                                  self._patch_number[0],
                                                  self._patch_number[1],
                                                  self._patch_size[0],
                                                  self._patch_size[1],
                                                  3])

        sub_images = tf.split(image, self._patch_number[0], axis=1)
        sub_images = reduce(lambda init_sub_images, sub_images: init_sub_images + tf.split(sub_images, self._patch_number[1], axis=2),
                            sub_images,
                            [])
        if batch_size != 1:
            sorted_sub_images = [[tf.squeeze(sub_image)
                                  for sub_image in sub_images[height_index * self._patch_number[0]: height_index * self._patch_number[0] + self._patch_number[0]]]
                                 for height_index in range(self._patch_number[1])]
        else:
            sorted_sub_images = [[tf.squeeze(sub_image)[tf.newaxis, ...]
                                  for sub_image in sub_images[height_index * self._patch_number[0]: height_index * self._patch_number[0] + self._patch_number[0]]]
                                 for height_index in range(self._patch_number[1])]

        image = tf.concat([tf.concat(images, axis=2) for images in sorted_sub_images], axis=1)
        return image


class ConvImageDecoder(ImageDecoderBlock):

    def __init__(self,
                 obj_name: str,
                 input_dimension: int,
                 output_dimensions: List[int],
                 patch_number: List[int],
                 is_training: Optional[bool],
                 training_phase: Optional[bool]):
        super().__init__(obj_name=obj_name,
                         input_dimension=input_dimension,
                         output_dimensions=output_dimensions,
                         is_training=is_training,
                         training_phase=training_phase)
        self._patch_number = patch_number

    def in_creating(self) -> NoReturn:
        self.conv_layers = []
        input_channel = self.input_dimension
        for i, output_channel in enumerate(self._output_dimensions):
            conv_name = f"conv_layer{i}"
            conv_layer = ConvolutionLayer.conv_bn_relu(obj_name=conv_name,
                                                       filter_size=[3, 3, input_channel, output_channel],
                                                       strides=[1, 1, 1, 1],
                                                       padding="SAME",
                                                       is_training=self.is_training)
            input_channel = output_channel
            self.conv_layers.append(conv_layer)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        if len(tf.shape(input_data)) == 3:
            batch_size, sequence_len, embedding_size = input_data.shape
            if self._patch_number[0] * self._patch_number[1] != sequence_len:
                raise ValueError(f"patch number {self._patch_number} not equal sequence length {sequence_len}")
            input_data = tf.reshape(input_data, shape=[batch_size, self._patch_number[0], self._patch_number[1], embedding_size])

        output = input_data
        for layer in self.conv_layers:
            output = layer.pipeline(input_data=output)

            _, height, width, _ = output.shape

            output = tf.image.resize(images=output, size=[2 * height, 2 * width])
        return output