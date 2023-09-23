from typing import List, Optional
from ospark.nn.model import Model
from ospark.nn.block import Block
from ospark.nn.block.image_decoder import DenseLayerDecoder
import tensorflow as tf
from ospark.nn.layers.dense_layer import DenseLayer
import ospark


class DiscreteVAE(Model):
    """
    DVAE model.
    """

    def __init__(self,
                 obj_name: str,
                 image_size: List[int],
                 patch_size: List[int],
                 token_number: int,
                 embedding_size: int,
                 temperature: float,
                 encoder: List[Block],
                 delay_create: Optional[bool]=None,
                 trained_weights: Optional[dict]=None,
                 decoder: Optional[List[Block]]=None,
                 is_training: Optional[bool]=None,
                 training_phase: Optional[bool]=None):
        """
        Args:
            obj_name: str
            image_size: List[int]
            patch_size: List[int]
            token_number: int
            embedding_size: int
            temperature: float
            encoder: List[Block]
            trained_weights: Optional[dict]
            decoder: Optional[List[Block]]
            is_training: Optional[bool]
            training_phase: Optional[bool]
        """

        super().__init__(obj_name=obj_name,
                         is_training=is_training,
                         delay_create=delay_create,
                         training_phase=training_phase,
                         trained_weights=trained_weights)
        self._image_size          = image_size
        self._patch_size          = patch_size
        self._height_patch_number = int(self._image_size[0] / self._patch_size[0])
        self._width_patch_number  = int(self._image_size[1] / self._patch_size[1])
        self._token_number        = token_number
        self._temperature         = temperature
        self._img_embed_weight    = ospark.weight_initializer.glorot_uniform(obj_name="img_embed_weight",
                                                                             shape=[token_number, embedding_size],
                                                                             trainable=is_training)

        self._classification_layer = DenseLayer(obj_name="classification_layer",
                                                input_dimension=embedding_size,
                                                hidden_dimension=[token_number],
                                                is_training=is_training)

        self._gumbel_sampling = -tf.math.log(-tf.math.log(tf.random.uniform(shape=[1,
                                                                                   self._height_patch_number * self._width_patch_number,
                                                                                   token_number], maxval=1)))
        self._encoder         = encoder

        if training_phase:
            self._decoder         = decoder or DenseLayerDecoder(obj_name="embedding_decoder",
                                                                input_dimension=embedding_size,
                                                                output_dimensions=[512, patch_size[0] * patch_size[1] * 3],
                                                                patch_size=patch_size,
                                                                patch_number=[self._height_patch_number,
                                                                              self._width_patch_number],
                                                                is_training=is_training)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        DVAE pipeline.

        Args:
            input_data: tf.Tensor

        Returns:
            output: tf.Tensor
        """

        output = self.tokenize(input_data=input_data)
        output = tf.nn.softmax(self.gumbel_softmax(input_data=output[:, 1:]))
        if self.training_phase:
            output = tf.matmul(output, self._img_embed_weight)
            output = self._decoder.pipeline(input_data=output)
        return output

    def gumbel_softmax(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        Gumbel softmax.

        Args:
            input_data: tf.Tensor

        Returns:
            gumbel_softmax: tf.Tensor
        """

        gumbel_softmax  = tf.math.log(tf.nn.softmax(input_data)) + self._gumbel_sampling
        gumbel_softmax  = tf.nn.softmax(gumbel_softmax / self._temperature)
        return gumbel_softmax

    def tokenize(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        Tokenize image embedding.

        Args:
            input_data: tf.Tensor

        Returns:
            tokens: tf.Tensor
        """

        encoder_output = self._encoder.pipeline(input_data=input_data)
        tokens         = self._classification_layer.pipeline(input_data=encoder_output)# [B, L, D] -> [B, L, N]
        return tokens


