from ospark import Model, weight_initializer, Layer, Block
from typing import Optional, List, Tuple
from ospark.nn.layers.self_attention import EncoderDecoderAttentionLayer, SelfAttentionLayer
from ospark.nn.layers.gated_linear_units import SwiGLU
from ospark.nn.layers.normalization import LayerNormalization
from ospark.nn.layers.feed_forward import FeedForwardLayer
from ospark.nn.layers.dense_layer import DenseLayer
from ospark.nn.block.transformer_block import TransformerDecoderBlock, TransformerEncoderBlock
from functools import reduce
import tensorflow as tf


class CoCa(Model):

    def __init__(self,
                 obj_name: str,
                 image_encoder: Block,
                 head_number: int,
                 embedding_size: int,
                 embedding_layer: Layer,
                 corpus_size: int,
                 pad_id: Optional[int]=None,
                 scale_rate: Optional[int]=None,
                 pooler: Optional[Layer]=None,
                 unimodal_decoder: Optional[List[Block]]=None,
                 mutimodal_decoder: Optional[List[Block]]=None,
                 mutimodal_block_number: Optional[int]=None,
                 unimodal_block_number: Optional[int]=None,
                 dropout_rate: Optional[float]=None,
                 image_queries_len: Optional[int]=None,
                 trained_weights: Optional[dict]=None,
                 is_training: Optional[bool]=None):
        super(CoCa, self).__init__(obj_name=obj_name, is_training=is_training, trained_weights=trained_weights)

        self._image_encoder     = image_encoder
        self._image_queries_len = image_queries_len or 256
        self._dropout_rate      = dropout_rate or 0.0
        self._embedding_size    = embedding_size
        self._head_number       = head_number
        self._scale_rate        = scale_rate or 4
        self._embedding_layer   = embedding_layer
        self._pad_id            = pad_id or 0
        self._cls_embedding     = weight_initializer.glorot_uniform(obj_name="cls_embedding",
                                                                    shape=[1, 1, self._embedding_size],
                                                                    trainable=is_training)
        self._layer_norm        = LayerNormalization(layer_dimension=embedding_size)

        self._unimodal_block_number  = unimodal_block_number or 2
        self._mutimodal_block_number = mutimodal_block_number or 2

        self._image_queries = weight_initializer.glorot_uniform(obj_name="image_queries",
                                                                shape=[1, self._image_queries_len + 1, embedding_size])

        self._att_pooling = pooler or EncoderDecoderAttentionLayer(obj_name="att_pooling",
                                                                   embedding_size=embedding_size,
                                                                   head_number=head_number,
                                                                   dropout_rate=dropout_rate,
                                                                   is_training=is_training)

        self._unimodal_decoder  = unimodal_decoder or self.create_unimodal(block_number=self._unimodal_block_number)
        self._mutimodal_decoder = mutimodal_decoder or self.create_mutimodal(block_number=self._mutimodal_block_number)

        self._output_layer      = Block(obj_name="logits_layer",
                                        layers=[LayerNormalization(layer_dimension=embedding_size),
                                                DenseLayer(obj_name="dense_layer",
                                                           input_dimension=embedding_size,
                                                           hidden_dimension=[corpus_size],
                                                           use_bias=False)])

    def create_unimodal(self, block_number: int) -> list:
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
            block = TransformerEncoderBlock(obj_name=f"unimodal_block_{i}",
                                            attention=att_layer,
                                            feedforward=ffnn_layer)
            decoder_blocks.append(block)
        return decoder_blocks

    def create_mutimodal(self, block_number: int) -> list:
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

    def pipeline(self, images: tf.Tensor, text: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_size      = len(text)
        image_feature   = self._image_encoder.pipeline(input_data=images)
        image_embedding = self._att_pooling.pipeline(input_data=tf.tile(self._image_queries, [batch_size, 1, 1]),
                                                         encoder_output=image_feature)
        cls_image_embedding, token_image_embedding = image_embedding[:, :1, :], image_embedding[:, 1:, :],

        padding_mask = tf.cast(tf.math.not_equal(text, self._pad_id), tf.float32)[..., tf.newaxis]  # [B, L]
        embeddings   = self._embedding_layer.pipeline(text) * padding_mask

        embeddings = tf.concat([embeddings, tf.tile(self._cls_embedding, [batch_size, 1, 1])], axis=-2)
        mask       = tf.cast(tf.math.equal(embeddings[..., 0], 0), tf.float32)[:, tf.newaxis, :]

        unimodal_output = reduce(lambda input_data, block: block.pipeline(input_data=input_data, mask=mask),
                                 self._unimodal_decoder,
                                 embeddings)

        token_embedding, cls_embedding = unimodal_output[:, :-1, :], unimodal_output[:, -1:, :]

        cls_embedding    = self._layer_norm(input_data=cls_embedding)
        mutimodal_output = reduce(lambda input_data, block: block.pipeline(input_data=input_data,
                                                                           encoder_output=token_image_embedding,
                                                                           encoder_padding_mask=None,
                                                                           decoder_padding_mask=mask[..., :-1]),
                                  self._mutimodal_decoder,
                                  token_embedding)

        logits = tf.nn.softmax(self._output_layer.pipeline(input_data=mutimodal_output))
        return logits, cls_image_embedding, cls_embedding

    def padding(self, input_data: List[tf.Tensor], lengths: List[int]) -> tf.Tensor:
        max_length = max(lengths)
        embeddings = [tf.pad(embedding[tf.newaxis, ...], [[0, 0], [0, max_length - lengths[i]], [0, 0]])
                      for i, embedding in enumerate(input_data)]
        return tf.concat(embeddings, axis=0)