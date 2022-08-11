from ospark.models.informer import Informer
from ospark.models.transformer import Transformer
from ospark.models.exdeep_transformer import ExdeepTransformer
from ospark.nn.block.informer_block import informer_decoder_block, informer_encoder_block
from ospark.nn.block.transformer_block import transformer_encoder_block, transformer_decoder_block, performer_decoder_block, performer_encoder_block
from ospark.nn.block.deep_transformer_block import exdeep_decoder_block, exdeep_encoder_block
from typing import Optional


class FormerBuilder:

    @staticmethod
    def informer(class_number: int,
                 block_number: int,
                 embedding_size: int,
                 head_number: int,
                 scale_rate: int,
                 sample_factor: float,
                 dropout_rate: float,
                 trained_weights: Optional[dict] = None,
                 is_training: Optional[bool] = None,
                 use_decoder: bool = True,
                 filter_width: int = None,
                 pooling_size: list = None,
                 strides: list = None
                 ) -> Informer:
            encoder_blocks = []
            decoder_blocks = []
            for i in range(block_number):
                encoder_name = f"encoder_block_{i}"
                encoder_blocks.append(informer_encoder_block(obj_name=encoder_name,
                                                             embedding_size=embedding_size,
                                                             head_number=head_number,
                                                             scale_rate=scale_rate,
                                                             sample_factor=sample_factor,
                                                             filter_width=filter_width,
                                                             pooling_size=pooling_size,
                                                             strides=strides,
                                                             is_training=is_training))
                if use_decoder:
                    decoder_name = f"decoder_block_{i}"
                    decoder_blocks.append(informer_decoder_block(obj_name=decoder_name,
                                                                 embedding_size=embedding_size,
                                                                 head_number=head_number,
                                                                 scale_rate=scale_rate,
                                                                 sample_factor=sample_factor,
                                                                 is_training=is_training))
            return Informer(obj_name="Informer",
                            trained_weights=trained_weights,
                            encoder_blocks=encoder_blocks,
                            class_number=class_number,
                            embedding_size=embedding_size,
                            decoder_blocks=decoder_blocks,
                            dropout_rate=dropout_rate,
                            is_training=is_training)

    @staticmethod
    def transformer(block_number: int,
                    head_number: int,
                    embedding_size: int,
                    scale_rate: int,
                    class_number: int,
                    dropout_rate: float,
                    trained_weights: Optional[dict] = None,
                    is_training: Optional[bool] = None,
                    encoder_corpus_size: Optional[int] = None,
                    decoder_corpus_size: Optional[int] = None,
                    use_embedding_layer: Optional[bool] = True,
                    max_length: Optional[int] = 2000,
                    ) -> Transformer:
        encoder_blocks = []
        decoder_blocks = []
        for i in range(block_number):
            encoder_name = f"encoder_block_{i}"
            encoder_blocks.append(transformer_encoder_block(obj_name=encoder_name,
                                                            embedding_size=embedding_size,
                                                            head_number=head_number,
                                                            scale_rate=scale_rate,
                                                            dropout_rate=dropout_rate,
                                                            is_training=is_training))
            decoder_name = f"decoder_block_{i}"
            decoder_blocks.append((transformer_decoder_block(obj_name=decoder_name,
                                                             embedding_size=embedding_size,
                                                             head_number=head_number,
                                                             scale_rate=scale_rate,
                                                             dropout_rate=dropout_rate,
                                                             is_training=is_training)))
        return Transformer(obj_name="Transformer",
                           trained_weights=trained_weights,
                           encoder_blocks=encoder_blocks,
                           class_number=class_number,
                           decoder_blocks=decoder_blocks,
                           max_length=max_length,
                           encoder_corpus_size=encoder_corpus_size,
                           decoder_corpus_size=decoder_corpus_size,
                           use_embedding_layer=use_embedding_layer,
                           embedding_size=embedding_size,
                           dropout_rate=dropout_rate,
                           is_training=is_training)

    @staticmethod
    def exdeep_transformer(encoder_block_number: int,
                           decoder_block_number: int,
                           head_number: int,
                           embedding_size: int,
                           scale_rate: int,
                           class_number: int,
                           dropout_rate: float,
                           trained_weights: Optional[dict] = None,
                           is_training: Optional[bool] = None,
                           encoder_corpus_size: Optional[int] = None,
                           decoder_corpus_size: Optional[int] = None,
                           use_embedding_layer: Optional[bool] = True,
                           max_length: Optional[int] = 2000,
                           ) -> Transformer:
        encoder_blocks = []
        decoder_blocks = []
        for i in range(encoder_block_number):
            encoder_name = f"encoder_block_{i}"
            encoder_blocks.append(exdeep_encoder_block(obj_name=encoder_name,
                                                       embedding_size=embedding_size,
                                                       head_number=head_number,
                                                       scale_rate=scale_rate,
                                                       dropout_rate=dropout_rate,
                                                       is_training=is_training,))
        for i in range(decoder_block_number):
            decoder_name = f"decoder_block_{i}"
            decoder_blocks.append((exdeep_decoder_block(obj_name=decoder_name,
                                                        embedding_size=embedding_size,
                                                        head_number=head_number,
                                                        scale_rate=scale_rate,
                                                        dropout_rate=dropout_rate,
                                                        is_training=is_training)))
        return ExdeepTransformer(obj_name="ExdeepTransformer",
                                 trained_weights=trained_weights,
                                 encoder_blocks=encoder_blocks,
                                 class_number=class_number,
                                 decoder_blocks=decoder_blocks,
                                 max_length=max_length,
                                 encoder_corpus_size=encoder_corpus_size,
                                 decoder_corpus_size=decoder_corpus_size,
                                 use_embedding_layer=use_embedding_layer,
                                 embedding_size=embedding_size,
                                 dropout_rate=dropout_rate,
                                 is_training=is_training)

    @staticmethod
    def performer(block_number: int,
                  head_number: int,
                  embedding_size: int,
                  scale_rate: int,
                  class_number: int,
                  dropout_rate: float,
                  random_projections_number: int,
                  trained_weights: Optional[dict] = None,
                  is_training: Optional[bool] = False,
                  encoder_corpus_size: Optional[int] = None,
                  decoder_corpus_size: Optional[int] = None,
                  use_embedding_layer: Optional[bool] = True,
                  max_length: Optional[int] = 2000,
                  ) -> Transformer:
        encoder_blocks = []
        decoder_blocks = []
        for i in range(block_number):
            encoder_name = f"encoder_block_{i}"
            encoder_blocks.append(performer_encoder_block(obj_name=encoder_name,
                                                          embedding_size=embedding_size,
                                                          head_number=head_number,
                                                          scale_rate=scale_rate,
                                                          dropout_rate=dropout_rate,
                                                          is_training=is_training,
                                                          random_projections_number=random_projections_number))
            decoder_name = f"decoder_block_{i}"
            decoder_blocks.append((performer_decoder_block(obj_name=decoder_name,
                                                           embedding_size=embedding_size,
                                                           head_number=head_number,
                                                           scale_rate=scale_rate,
                                                           dropout_rate=dropout_rate,
                                                           is_training=is_training,
                                                           random_projections_number=random_projections_number)))
        return Transformer(obj_name="Transformer",
                           trained_weights=trained_weights,
                           encoder_blocks=encoder_blocks,
                           class_number=class_number,
                           decoder_blocks=decoder_blocks,
                           max_length=max_length,
                           encoder_corpus_size=encoder_corpus_size,
                           decoder_corpus_size=decoder_corpus_size,
                           use_embedding_layer=use_embedding_layer,
                           embedding_size=embedding_size,
                           dropout_rate=dropout_rate,
                           is_training=is_training)
