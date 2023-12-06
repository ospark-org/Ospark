from ospark.trainer import Trainer
from ospark.nn.block.image_decoder import ConvImageDecoder
from ospark.backbone.discrete_vae import DiscreteVAE
from ospark.backbone.vision_transformer import VisionTransformer
from ospark.data.generator.dvae_data_generator import DVAEDataGenerator
from ospark.nn.optimizer.learning_rate_schedule import TransformerWarmup
from ospark.data.processor.auto_click import AutoClick
from typing import List, Optional
import tensorflow as tf
import json

#folder_path = "/Volumes/T7/23.02.20/標註/D_手開三聯式/完成/img(DA)"
# folder_path = "/Volumes/T7/42651975_意德/112/5-6"


class DVAECommandTrainer(metaclass=AutoClick):

    def __init__(self,
                 train_data_folder: str,
                 train_epoch_number: int,
                 batch_size: int,
                 resize_target: List[int],
                 image_size: List[int],
                 patch_size: List[int],
                 head_number: int,
                 scale_rate: int,
                 dropout_rate: float,
                 embedding_size: int,
                 encoder_number: int,
                 save_weights_path: str,
                 save_info_path: str,
                 load_weights_path: Optional[str],
                 temperature: Optional[float],
                 token_number: Optional[int],
                 conv_output_dimensions: Optional[List[int]],
                 patch_number: Optional[List[int]],
                 warmup_step: Optional[float],
                 beta_1: Optional[float],
                 beta_2: Optional[float],
                 epsilon: Optional[float]):
        self._warmup_step  = warmup_step or 4000.
        self._beta_1       = beta_1 or 0.9
        self._beta_2       = beta_2 or 0.98
        self._epsilon      = epsilon or 1e-9
        self._patch_number = patch_number or [32, 32]
        self._token_number = token_number or 8192
        self._temperature  = temperature or 0.9

        self._conv_output_dimensions = conv_output_dimensions or [128, 64, 3]


    def main_process(self):
        if self._load_weights_path is not None:
            with open(self._load_weights_path, 'r') as fp:
                weights = json.load(fp)
        else:
            weights = None

        encoder = VisionTransformer(obj_name="image_encoder",
                                    image_height=self._image_size[0],
                                    image_width=self._image_size[1],
                                    patch_height=self._patch_size[0],
                                    patch_width=self._patch_size[1],
                                    head_number=self._head_number,
                                    scale_rate=self._scale_rate,
                                    dropout_rate=self._dropout_rate,
                                    embedding_size=self._embedding_size,
                                    encoder_number=self._encoder_number,
                                    is_training=True)

        decoder = ConvImageDecoder(obj_name="ConvImageDecoder",
                                   input_dimension=self._embedding_size,
                                   output_dimensions=self._conv_output_dimensions,
                                   patch_number=self._patch_number,
                                   is_training=True,
                                   training_phase=True)

        model = DiscreteVAE(obj_name="DVAE",
                           image_size=self._image_size,
                           patch_size=self._patch_size,
                           token_number=self._token_number,
                           embedding_size=self._embedding_size,
                           temperature=self._temperature,
                           encoder=encoder,
                           decoder=decoder,
                           training_phase=True,
                           trained_weights=weights,
                           is_training=True)

        data_generator = DVAEDataGenerator(train_data_folder=self._train_data_folder,
                                           batch_size=self._batch_size,
                                           resize_target=self._resize_target)

        loss_function = lambda prediction, target_data: tf.reduce_mean(tf.pow(prediction - target_data/100, 2))
        learning_rate = TransformerWarmup(model_dimension=self._embedding_size, warmup_step=self._warmup_step)
        optimizer     = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate,
                                                        beta_1=self._beta_1,
                                                        beta_2=self._beta_2,
                                                        epsilon=self._epsilon)

        trainer = Trainer(model=model,
                          data_generator=data_generator,
                          epoch_number=self._train_epoch_number,
                          optimizer=optimizer,
                          loss_function=loss_function,
                          use_auto_graph=True,
                          save_weights_path=self._save_weights_path,
                          save_info_path=self._save_info_path,
                          presentation_of_loss_value=100,
                          save_times=5)
        trainer.start()


if __name__ == "__main__":
    DVAECommandTrainer()