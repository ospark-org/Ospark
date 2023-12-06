from ospark.nn.model import Model
from typing import List, Union, Tuple
from ospark.backbone.vision_transformer import VisionTransformer
from ospark.algorithm.blokcwise_making import BlockwiseMasking
from ospark.nn.layers.dense_layer import DenseLayer
from typing import Optional
import tensorflow as tf


class BEiT(Model):
    """
    BEiT model. paper url: https://arxiv.org/pdf/2106.08254.pdf
    """

    def __init__(self,
                 obj_name: str,
                 image_size: List[int],
                 patch_size: List[int],
                 block_number: int,
                 head_number: int,
                 embedding_size: int,
                 scale_rate: int,
                 delay_create: Optional[bool]=None,
                 corpus_size: Optional[int]=None,
                 trained_weights: Optional[dict]=None,
                 is_training: Optional[bool]=None,
                 dropout_rate: Optional[float]=None,
                 use_classify_layer: Optional[bool]=None,
                 image_encoder: Optional[Model]=None,
                 training_phase: Optional[bool]=None):
        """
        Args:
            obj_name: str
            image_size: List[int]
            patch_size: List[int]
            block_number: int
            head_number: int
            embedding_size: int
            scale_rate: int
            delay_create: Optional[bool]
            corpus_size: Optional[int]
            trained_weights: Optional[dict]
            is_training: Optional[bool]
            dropout_rate: Optional[float]
            use_classify_layer: Optional[bool]
            image_encoder: Optional[Model]
            training_phase: Optional[bool]
        """

        super().__init__(obj_name=obj_name,
                         delay_create=delay_create,
                         training_phase=training_phase,
                         is_training=is_training,
                         trained_weights=trained_weights)
        self._image_size     = image_size
        self._patch_size     = patch_size
        self._block_number   = block_number
        self._head_number    = head_number
        self._embedding_size = embedding_size
        self._scale_rate     = scale_rate
        self._training_phase = training_phase
        self._corpus_size    = corpus_size
        self._dropout_rate   = dropout_rate

        self._image_encoder = image_encoder or VisionTransformer(obj_name="image_encoder",
                                                                 delay_create=delay_create,
                                                                 image_height=image_size[0],
                                                                 image_width=image_size[1],
                                                                 patch_height=patch_size[0],
                                                                 patch_width=patch_size[1],
                                                                 head_number=head_number,
                                                                 encoder_number=block_number,
                                                                 scale_rate=scale_rate,
                                                                 dropout_rate=dropout_rate,
                                                                 embedding_size=embedding_size)


        self._use_classify_layer = use_classify_layer if use_classify_layer is not None else False

        if use_classify_layer:
            self._output_layer = DenseLayer(obj_name="output_layer",
                                            input_dimension=embedding_size,
                                            hidden_dimension=[corpus_size])

    def pipeline(self, input_data: tf.Tensor, mask_matrices: Optional[tf.Tensor]=None) -> tf.Tensor:
        """
        Model pipeline.

        Args:
            input_data: tf.Tensor
            mask_matrices: Optional[tf.Tensor]

        Returns:
            output: tf.Tensor
        """

        output = self._image_encoder.pipeline(input_data=input_data, mask_matrix=mask_matrices)
        if self._use_classify_layer:
            output = self._output_layer.pipeline(input_data=output[:, 1:, ...])
            output = output * tf.cast(tf.math.equal(mask_matrices, 0), dtype=tf.float32) if mask_matrices is not None else tf.nn.softmax(output)
        return output
