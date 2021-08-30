from ospark.backbone.auto_encoder import AutoEncoder
from ospark.backbone.backbone import Backbone
from ospark.backbone.resnet import resnet_50
from ospark.nn.block import Block
from ospark.nn.block.connection_block import shared_convolution_decoder
import tensorflow as tf

class Unet(AutoEncoder):

    def __init__(self,
                 obj_name: str,
                 encoder: Backbone,
                 decoder: Block):
        super().__init__(obj_name=obj_name,
                         encoder=encoder,
                         decoder=decoder)

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        encoder_output = self.encoder(input_data)
        decoder_output = self.decoder(encoder_output, self.encoder.catch_box)
        return decoder_output

def shared_convolution(trainable: bool) -> Unet:
    encoder = resnet_50(trainable=trainable, catch_output=True)
    input_channels = [2048 + 1024, 128 + 512, 64 + 256]
    output_channels = [128, 64, 32]
    decoder = shared_convolution_decoder(input_channels=input_channels,
                                         output_channels=output_channels,
                                         trainable=trainable)
    shared_convolution = Unet(obj_name="shared_convolution",
                              encoder=encoder,
                              decoder=decoder)
    return shared_convolution

