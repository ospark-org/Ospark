from ospark.backbone.vision_transformer import VisionTransformer
from ospark.backbone.discrete_vae import DiscreteVAE
from ospark.trainer.beit_trainer import BEiTTrainer
from typing import Optional, List
from ospark.data.generator.dvae_data_generator import DVAEDataGenerator
from ospark.nn.optimizer.learning_rate_schedule import TransformerWarmup
from ospark.nn.loss_function.sparse_categorical_cross_entropy import SparseCategoricalCrossEntropy
from ospark.models.BEiT import BEiT
from ospark.data.processor.auto_click import AutoClick
import tensorflow as tf
import json


class BEiTCommandTrainer(metaclass=AutoClick):

    def __init__(self,
                 folder_path: str,
                 image_size: List[int],
                 patch_size: List[int],
                 batch_size: int,
                 embedding_size: int,
                 resize_target: List[int],
                 head_number: int,
                 scale_rate: int,
                 encoder_number: int,
                 beit_encoder_number: int,
                 train_epoch_number: int,
                 dvae_weights_path: str,
                 weight_save_path: str,
                 info_save_path: str,
                 weight_load_path: Optional[str]=None,
                 info_load_path: Optional[str]=None,
                 dropout_rate: Optional[float]=None,
                 token_number: Optional[int]=None,
                 temperature: Optional[float]=None,
                 use_auto_graph: Optional[bool]=None,
                 warmup_step: Optional[float]=None,
                 beta_one: Optional[float]=None,
                 beta_two: Optional[float]=None,
                 epsilon: Optional[float]=None):
        self._token_number   = token_number or 8192
        self._temperature    = temperature or 0.9
        self._use_auto_graph = use_auto_graph if use_auto_graph is not None else True
        self._warmup_step    = warmup_step or 4000.
        self._dropout_rate   = dropout_rate or 0.1

        self._beta_one = beta_one or 0.9
        self._beta_two = beta_two or 0.98
        self._epsilon  = epsilon or 1e-9

    def main_process(self):
        data_generator = DVAEDataGenerator(train_data_folder=self._folder_path,
                                           batch_size=self._batch_size,
                                           resize_target=self._resize_target)
        loss_function  = SparseCategoricalCrossEntropy()
        learning_rate  = TransformerWarmup(model_dimension=self._embedding_size, warmup_step=self._warmup_step)
        optimizer      = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate,
                                                         beta_1=self._beta_one,
                                                         beta_2=self._beta_two,
                                                         epsilon=self._epsilon)

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
                                    is_training=False,
                                    delay_create=True)

        with open(self._dvae_weights_path, 'r') as fp:
            dvae_weights = json.load(fp)

        tokenizer = DiscreteVAE(obj_name="DVAE",
                                image_size=self._image_size,
                                patch_size=self._patch_size,
                                token_number=self._token_number,
                                embedding_size=self._embedding_size,
                                temperature=self._temperature,
                                encoder=encoder,
                                training_phase=False,
                                trained_weights=dvae_weights,
                                is_training=False)

        beit_weights = None
        if self._weight_load_path is not None:
            with open(self._weight_load_path, 'r') as fp:
                beit_weights = json.load(fp)

        model = BEiT(obj_name="beit",
                     image_size=self._image_size,
                     patch_size=self._patch_size,
                     head_number=self._head_number,
                     scale_rate=self._scale_rate,
                     dropout_rate=self._dropout_rate,
                     embedding_size=self._embedding_size,
                     block_number=self._beit_encoder_number,
                     use_classify_layer=True,
                     corpus_size=self._token_number,
                     trained_weights=beit_weights)

        trainer = BEiTTrainer(model=model,
                              tokenizer=tokenizer,
                              data_generator=data_generator,
                              epoch_number=self._train_epoch_number,
                              optimizer=optimizer,
                              loss_function=loss_function,
                              save_weights_path=self._weight_save_path,
                              save_info_path=self._info_save_path,
                              use_auto_graph=False)

        trainer.start()


if __name__ == "__main__":
    BEiTCommandTrainer()