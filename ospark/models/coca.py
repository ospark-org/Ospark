from ospark import Model, weight_initializer, Layer, Block
from typing import Optional, List, Tuple, Union
from ospark.nn.layers.self_attention import EncoderDecoderAttentionLayer, SelfAttentionLayer
from ospark.nn.layers.gated_linear_units import SwiGLU
from ospark.nn.layers.normalization import LayerNormalization
from ospark.nn.layers.feed_forward import FeedForwardLayer
from ospark.nn.layers.dense_layer import DenseLayer
from ospark.nn.block.transformer_block import TransformerDecoderBlock, TransformerEncoderBlock
from functools import reduce
import tensorflow as tf
import numpy as np


class CoCa(Model):
    """
    CoCa Model. paper url: https://arxiv.org/pdf/2205.01917.pdf
    """

    def __init__(self,
                 obj_name: str,
                 image_encoder: Block,
                 head_number: int,
                 embedding_size: int,
                 embedding_layer: Layer,
                 corpus: dict,
                 delay_create: Optional[bool]=None,
                 use_predict_result: Optional[bool]=None,
                 pad_id: Optional[int]=None,
                 scale_rate: Optional[int]=None,
                 att_pooling: Optional[Layer]=None,
                 unimodal_decoder: Optional[List[Block]]=None,
                 mutimodal_decoder: Optional[List[Block]]=None,
                 mutimodal_block_number: Optional[int]=None,
                 unimodal_block_number: Optional[int]=None,
                 dropout_rate: Optional[float]=None,
                 image_queries_len: Optional[int]=None,
                 trained_weights: Optional[dict]=None,
                 is_training: Optional[bool]=None,
                 training_phase: Optional[bool]=None):
        """
        Args:
            obj_name: str
            image_encoder: Block
            head_number: int
            embedding_size: int
            embedding_layer: Layer
            corpus: dict
            delay_create: Optional[bool]
            use_predict_result: Optional[bool]
            pad_id: Optional[int]
            scale_rate: Optional[int]
            att_pooling: Optional[Layer]
            unimodal_decoder: Optional[List[Block]]
            mutimodal_decoder: Optional[List[Block]]
            mutimodal_block_number: Optional[int]
            unimodal_block_number: Optional[int]
            dropout_rate: Optional[float]
            image_queries_len: Optional[int]
            trained_weights: Optional[dict]
            is_training: Optional[bool]
            training_phase: Optional[bool]
        """

        super(CoCa, self).__init__(obj_name=obj_name,
                                   delay_create=delay_create,
                                   is_training=is_training,
                                   trained_weights=trained_weights,
                                   training_phase=training_phase)

        self._image_encoder      = image_encoder
        self._image_queries_len  = image_queries_len or 256
        self._dropout_rate       = dropout_rate or 0.0
        self._embedding_size     = embedding_size
        self._head_number        = head_number
        self._scale_rate         = scale_rate or 4
        self._embedding_layer    = embedding_layer
        self._pad_id             = pad_id or 0
        self._max_length         = 1000
        self._corpus             = corpus
        self._index_to_char      = {value: key for key, value in corpus.items()}
        self._use_predict_result = use_predict_result
        self._cls_embedding      = weight_initializer.glorot_uniform(obj_name="cls_embedding",
                                                                     shape=[1, 1, self._embedding_size],
                                                                     trainable=is_training)
        self._layer_norm         = LayerNormalization(layer_dimension=embedding_size)

        self._unimodal_block_number  = unimodal_block_number or 2
        self._mutimodal_block_number = mutimodal_block_number or 2

        self._image_queries = weight_initializer.glorot_uniform(obj_name="image_queries",
                                                                shape=[1, self._image_queries_len + 1, embedding_size])

        self._att_pooling = att_pooling or EncoderDecoderAttentionLayer(obj_name="att_pooling",
                                                                        embedding_size=embedding_size,
                                                                        head_number=head_number,
                                                                        dropout_rate=self._dropout_rate,
                                                                        is_training=is_training)

        self._unimodal_decoder  = unimodal_decoder or self.create_unimodal(block_number=self._unimodal_block_number)
        self._mutimodal_decoder = mutimodal_decoder or self.create_mutimodal(block_number=self._mutimodal_block_number)

        self._output_layer      = Block(obj_name="logits_layer",
                                        layers=[LayerNormalization(layer_dimension=embedding_size),
                                                DenseLayer(obj_name="dense_layer",
                                                           input_dimension=embedding_size,
                                                           hidden_dimension=[len(corpus)],
                                                           use_bias=False)])

        self._positional_embedding = self.create_positional_encoding_table()

    def create_positional_encoding_table(self) -> tf.Tensor:
        basic_table = np.zeros(shape=[self._max_length, self._embedding_size])
        position = np.arange(self._max_length).reshape([-1, 1])
        denominator = np.power(10000, np.arange(0, self._embedding_size, 2) / self._embedding_size)
        basic_table[:, 0::2] = np.sin(position / denominator)
        basic_table[:, 1::2] = np.cos(position / denominator)
        return tf.convert_to_tensor(basic_table, dtype=tf.float32)[tf.newaxis, :, :]

    def create_unimodal(self, block_number: int) -> List[Block]:
        """
        Create uni-modal.

        Args:
            block_number: int

        Returns:
            decoder_blocks: List[Block]
        """

        decoder_blocks = []
        for i in range(block_number):
            att_layer = SelfAttentionLayer(obj_name="att",
                                           embedding_size=self._embedding_size,
                                           head_number=self._head_number,
                                           dropout_rate=self._dropout_rate,
                                           use_look_ahead=True)
            ffnn_layer = FeedForwardLayer(obj_name="ffnn",
                                          embedding_size=self._embedding_size,
                                          scale_rate=self._scale_rate,
                                          dropout_rate=self._dropout_rate,
                                          activation=SwiGLU(dimension=self._embedding_size * self._scale_rate,
                                                            is_training=self.is_training))
            block = TransformerEncoderBlock(obj_name=f"unimodal_block_{i}", feedforward=ffnn_layer, attention=att_layer)
            decoder_blocks.append(block)
        return decoder_blocks

    def create_mutimodal(self, block_number: int) -> List[Block]:
        """
        Create muti-modal.

        Args:
            block_number: int

        Returns:
            decoder_blocks: List[Block]
        """

        decoder_blocks = []
        for i in range(block_number):
            att = SelfAttentionLayer(obj_name="att",
                                     embedding_size=self._embedding_size,
                                     head_number=self._head_number,
                                     dropout_rate=self._dropout_rate,
                                     use_look_ahead=True,
                                     is_training=self.is_training)
            encode_decode_att = EncoderDecoderAttentionLayer(obj_name="encode_decode_att",
                                                             embedding_size=self._embedding_size,
                                                             head_number=self._head_number,
                                                             dropout_rate=self._dropout_rate,
                                                             is_training=self.is_training)

            ffnn_layer = FeedForwardLayer(obj_name="ffnn",
                                          embedding_size=self._embedding_size,
                                          scale_rate=self._scale_rate,
                                          dropout_rate=self._dropout_rate,
                                          activation=SwiGLU(dimension=self._embedding_size * self._scale_rate,
                                                            is_training=self.is_training),
                                          is_training=self.is_training)

            block = TransformerDecoderBlock(obj_name=f"mutimodal{i}",
                                            attention=att,
                                            encode_decode_attention=encode_decode_att,
                                            feedforward=ffnn_layer)
            decoder_blocks.append(block)
        return decoder_blocks

    def pipeline(self, images: tf.Tensor, text: Optional[tf.Tensor]=None) -> Union[str, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        """
        Model pipeline.

        Args:
            images: tf.Tensor
            text: Optional[tf.Tensor]

        Returns:
            result: Union[str, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]
        """

        batch_size = images.shape[0]

        image_feature = self._image_encoder.pipeline(input_data=images)

        image_embedding = self._att_pooling.pipeline(input_data=tf.tile(self._image_queries, [batch_size, 1, 1]),
                                                     encoder_output=image_feature)

        cls_image_embedding, token_image_embedding = image_embedding[:, :1, :], image_embedding[:, 1:, :],

        if text is None:
            init_char = [self._corpus["BOS"]]
            eos_index = self._corpus["EOS"]
            while init_char[-1] != eos_index:
                embeddings = self._embedding_layer.pipeline(init_char)[tf.newaxis, ...] + self._positional_embedding[:, :len(init_char), :]
                mask       = None
                embeddings = tf.concat([embeddings, tf.tile(self._cls_embedding, [batch_size, 1, 1])], axis=-2)

                logits, cls_embedding = self.infer(embeddings=embeddings,
                                                   mask=mask,
                                                   token_image_embedding=token_image_embedding)
                arg_index  = tf.argmax(logits, axis=-1).numpy()
                init_char.append(arg_index[0][-1])
            return "".join([self._index_to_char[index] for index in init_char[1:-1]])

        else:
            padding_mask = tf.cast(tf.math.not_equal(text, self._pad_id), tf.float32)[..., tf.newaxis]  # [B, L]
            embeddings   = (self._embedding_layer.pipeline(text) + self._positional_embedding[:, :text.shape[1], :]) * padding_mask

            embeddings = tf.concat([embeddings, tf.tile(self._cls_embedding, [batch_size, 1, 1])], axis=-2)
            mask       = tf.cast(tf.math.equal(embeddings[..., 0], 0), tf.float32)[:, tf.newaxis, :]

            logits, cls_embedding  = self.infer(embeddings=embeddings,
                                                mask=mask,
                                                token_image_embedding=token_image_embedding)
            return logits, cls_image_embedding, cls_embedding


    def padding(self, input_data: List[tf.Tensor], lengths: List[int]) -> tf.Tensor:
        """
        Padding zero.

        Args:
            input_data: List[tf.Tensor]
            lengths: List[int]

        Returns:
            result: tf.Tensor
        """

        max_length = max(lengths)
        embeddings = [tf.pad(embedding[tf.newaxis, ...], [[0, 0], [0, max_length - lengths[i]], [0, 0]])
                      for i, embedding in enumerate(input_data)]
        return tf.concat(embeddings, axis=0)

    def infer(self, embeddings: tf.Tensor, mask: Union[tf.Tensor, None], token_image_embedding: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Inference.

        Args:
            embeddings: tf.Tensor
            mask: Union[tf.Tensor, None]
            token_image_embedding: tf.Tensor

        Returns:
            logits: tf.Tensor
            cls_embedding: tf.Tensor
        """

        unimodal_output = reduce(lambda input_data, block: block.pipeline(input_data=input_data, mask=mask),
                                 self._unimodal_decoder,
                                 embeddings)

        token_embedding, cls_embedding = unimodal_output[:, :-1, :], unimodal_output[:, -1:, :]

        cls_embedding    = self._layer_norm(input_data=cls_embedding)

        decoder_padding_mask = mask[..., :-1] if mask is not None else None

        mutimodal_output = reduce(lambda input_data, block: block.pipeline(input_data=input_data,
                                                                           encoder_output=token_image_embedding,
                                                                           encoder_padding_mask=None,
                                                                           decoder_padding_mask=decoder_padding_mask),
                                  self._mutimodal_decoder,
                                  token_embedding)

        logits = tf.nn.softmax(self._output_layer.pipeline(input_data=mutimodal_output))
        return logits, cls_embedding