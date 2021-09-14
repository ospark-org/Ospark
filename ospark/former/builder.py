from ospark.former.informer import Informer
from ospark.former.transformer import Transformer
from ospark.nn.block.informer_block import informer_decoder_block, informer_encoder_block
from ospark.nn.block.transformer_block import transformer_encoder_block, transformer_decoder_block, exdeep_decoder_block, exdeep_encoder_block
from ospark.nn.component.normalization import Normalization
from typing import Optional


def build_informer(class_number: int,
                   block_number: int,
                   embedding_size: int,
                   head_number: int,
                   scale_rate: int,
                   sample_factor: float,
                   use_decoder: bool=True,
                   use_graph_mode: bool=True,
                   filter_width: int=None,
                   pooling_size: list=None,
                   strides: list=None
                   ) -> Informer:
        encoder_blocks = []
        decoder_blocks = []
        for i in range(block_number):
            encoder_name = f"encoder_block_{i}"
            encoder_blocks.append(informer_encoder_block(encoder_name, embedding_size, head_number, scale_rate, sample_factor, filter_width, pooling_size, strides))
            if use_decoder:
                decoder_name = f"decoder_block_{i}"
                decoder_blocks.append(informer_decoder_block(decoder_name, embedding_size, head_number, scale_rate, sample_factor))
        return Informer("Informer", encoder_blocks, class_number, embedding_size, use_graph_mode, decoder_blocks)


def build_transformer(block_number: int,
                     head_number: int,
                     embedding_size :int,
                     scale_rate: int,
                     class_number :int,
                     max_length: int=2000,
                     normalization: Optional[Normalization]=None,
                     initial_norm: bool=False) -> Transformer:
    encoder_blocks = []
    decoder_blocks = []
    for i in range(block_number):
        encoder_name = f"encoder_block_{i}"
        encoder_blocks.append(transformer_encoder_block(encoder_name, embedding_size, head_number, scale_rate))
        decoder_name = f"decoder_block_{i}"
        decoder_blocks.append((transformer_decoder_block(decoder_name, embedding_size, head_number, scale_rate)))
    return Transformer(obj_name="Transformer", encoder_blocks=encoder_blocks, class_number=class_number, decoder_blocks=decoder_blocks,
                       max_length=max_length, normalization=normalization, initial_norm=initial_norm, embedding_size=embedding_size)


def build_exdeep_transformer(encoder_block_number: int,
                             decoder_block_number: int,
                             head_number: int,
                             embedding_size: int,
                             scale_rate: int,
                             class_number: int,
                             max_length: int=2000,
                             normalization: Optional[Normalization]=None,
                             initial_norm: bool=False) -> Transformer:
    encoder_blocks = []
    decoder_blocks = []
    for i in range(encoder_block_number):
        encoder_name = f"encoder_block_{i}"
        encoder_blocks.append(exdeep_encoder_block(encoder_name, embedding_size, head_number, scale_rate))
    for i in range(decoder_block_number):
        decoder_name = f"decoder_block_{i}"
        decoder_blocks.append((exdeep_decoder_block(decoder_name, embedding_size, head_number, scale_rate)))
    return Transformer(obj_name="ExdeepTransformer", encoder_blocks=encoder_blocks, class_number=class_number, decoder_blocks=decoder_blocks,
                       max_length=max_length, normalization=normalization, initial_norm=initial_norm, embedding_size=embedding_size)
