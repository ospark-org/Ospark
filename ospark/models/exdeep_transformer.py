from ospark.models.transformer import *
from ospark.nn.block.deep_transformer_block import DeepTransformerEncoder, DeepTransformerDecoder


class ExdeepTransformer(Transformer):

    def __init__(self,
                 obj_name: str,
                 encoder_blocks: List[DeepTransformerEncoder],
                 class_number: int,
                 embedding_size: int,
                 dropout_rate: float,
                 trained_weights: Optional[dict]=None,
                 is_training: Optional[bool]=False,
                 encoder_corpus_size: Optional[int]=None,
                 decoder_corpus_size: Optional[int]=None,
                 use_embedding_layer: Optional[bool]=True,
                 use_classifier: Optional[bool]=False,
                 decoder_blocks: Optional[List[DeepTransformerDecoder]]=None,
                 max_length: int=2000,
                 ) -> NoReturn:
        super().__init__(obj_name=obj_name,
                         trained_weights=trained_weights,
                         encoder_blocks=encoder_blocks,
                         class_number=class_number,
                         embedding_size=embedding_size,
                         use_embedding_layer=use_embedding_layer,
                         encoder_corpus_size=encoder_corpus_size,
                         decoder_corpus_size=decoder_corpus_size,
                         use_classifier=use_classifier,
                         decoder_blocks=decoder_blocks,
                         max_length=max_length,
                         dropout_rate=dropout_rate,
                         is_training=is_training
                         )

    @property
    def encoder_blocks(self) -> List[DeepTransformerEncoder]:
        return self._encoder_blocks

    @property
    def decoder_blocks(self) -> List[DeepTransformerDecoder]:
        return self._decoder_blocks


    def profiling_phase(self, encoder_input: tf.Tensor, decoder_input) -> NoReturn:
        catch_list = []
        seq_len        = tf.shape(encoder_input)[1]
        encoder_padding_mask, lookahead_mask = self.create_mask_matrix(encoder_input, decoder_input)

        encoder_input  = self.encoder_embedding_layer.pipeline(encoder_input)
        encoder_input *= self.embedding_scale_rate
        encoder_input += self.positional_encoding_table[:, seq_len, :]
        encoder_input  = self.encoder_dropout_layer(encoder_input, training=self.is_training)
        output = encoder_input
        for encoder_block in self.encoder_blocks:
            output, catch_list = encoder_block.profiling_phase(input_data=output,
                                                               catch_variance=catch_list,
                                                               mask=encoder_padding_mask)

        if self.decoder_blocks != []:
            seq_len        = tf.shape(encoder_input)[1]
            decoder_input  = self.decoder_embedding_layer(decoder_input)
            decoder_input *= self.embedding_scale_rate
            decoder_input += self.positional_encoding_table[:, seq_len, :]
            decoder_input  = self.decoder_dropout_layer(decoder_input, training=self.is_training)
            encoder_output = output
            output = decoder_input
            for decoder_block in self.decoder_blocks:
                output, catch_list = decoder_block.profiling_phase(input_data=output,
                                                                   encoder_output=encoder_output,
                                                                   catch_variance=catch_list,
                                                                   encoder_padding_mask=encoder_padding_mask,
                                                                   decoder_padding_mask=lookahead_mask)

    def back_to_standard(self) -> NoReturn:
        _ = [encoder_block.recover_transformer() for encoder_block in self.encoder_blocks]
        if self.decoder_blocks != []:
            _ = [decoder_block.recover_transformer() for decoder_block in self.decoder_blocks]
