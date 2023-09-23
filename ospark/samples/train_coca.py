import json
import tensorflow as tf
from ospark.nn.model import Model
from ospark.models.coca import CoCa
from ospark.data.generator.coca_data_generator import CoCaDateGenerator
from ospark.trainer.coca_trainer import CocaTrainer
from ospark.models.BEiT import BEiT
from ospark.nn.layers.embedding_layer import EmbeddingLayer
from ospark.nn.loss_function.sparse_categorical_cross_entropy import SparseCategoricalCrossEntropy
from ospark.nn.optimizer.learning_rate_schedule import TransformerWarmup
from ospark.data.processor.auto_click import AutoClick
from typing import Optional, List


class CoCaCommandTrainer(metaclass=AutoClick):
    """
    CoCa trainer.
    """

    def __init__(self,
                 epoch_number: int,
                 image_size: List[int],
                 patch_size: List[int],
                 block_number: int,
                 head_number: int,
                 embedding_size: int,
                 scale_rate: int,
                 dropout_rate: float,
                 batch_size: int,
                 weights_save_path: str,
                 info_save_path: str,
                 training_data_path: str,
                 corpus_path: str,
                 weights_load_path: Optional[str],
                 info_load_path: Optional[str],
                 use_auto_graph: Optional[bool],
                 warmup_step: Optional[float]):
        """
        Args:
            epoch_number: int
            image_size: str
            patch_size: str
            block_number: int
            head_number: int
            embedding_size: int
            scale_rate: int
            dropout_rate: float
            batch_size: int
            weights_save_path: str
            info_save_path: str
            corpus_path: str
            training_data_path: str
            use_auto_graph: Optional[bool]
            weights_load_path: Optional[str]
            info_load_path: Optional[str]
            warmup_step: Optional[float]
        """

        self._warmup_step  = warmup_step or 4000.
        self._use_auto_graph = use_auto_graph if use_auto_graph is not None else True

    def main_process(self):
        with open(self._corpus_path, 'r') as fp:
            corpus = json.load(fp)

        coca_weights = None
        try:
            with open(self._weights_load_path, 'r') as fp:
                coca_weights = json.load(fp)
        except:
            print(f"weights path: {self._weights_load_path} is not exist.")

        if self._info_load_path is None:
            image_encoder = BEiT(obj_name="image_encoder",
                                 image_size=self._image_size,
                                 patch_size=self._patch_size,
                                 block_number=self._block_number,
                                 head_number=self._head_number,
                                 embedding_size=self._embedding_size,
                                 scale_rate=self._scale_rate,
                                 dropout_rate=self._dropout_rate,
                                 is_training=True,
                                 delay_create=True)

            embedding_layer = EmbeddingLayer(obj_name="embedding_layer",
                                             embedding_dimension=self._embedding_size,
                                             corpus_size=len(corpus))

            model = CoCa(obj_name="coca",
                         image_encoder=image_encoder,
                         head_number=self._head_number,
                         embedding_size=self._embedding_size,
                         embedding_layer=embedding_layer,
                         corpus=corpus,
                         scale_rate=self._scale_rate,
                         dropout_rate=self._dropout_rate,
                         trained_weights=None,
                         use_predict_result=True)
        else:
            with open(self._load_info_path, 'r') as fp:
                info = json.load(fp)

            model = Model.create_from_info(info, trained_weights=coca_weights)

        data_generator = CoCaDateGenerator(training_dataset_path=self._training_data_path,
                                           corpus_data_or_path=corpus,
                                           batch_size=self._batch_size,
                                           resize_target=self._image_size)

        learning_rate  = TransformerWarmup(model_dimension=self._embedding_size,
                                           warmup_step=self._warmup_step)
        optimizer      = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate,
                                                         beta_1=0.9,
                                                         beta_2=0.98,
                                                         epsilon=1e-9)
        loss_function  = {"contrastive_loss": lambda prediction, target_data: tf.reduce_mean(tf.pow(prediction - target_data/100, 2)),
                          "caption_loss": SparseCategoricalCrossEntropy(ignore_class=0)}

        trainer = CocaTrainer(model=model,
                              data_generator=data_generator,
                              epoch_number=self._epoch_number,
                              optimizer=optimizer,
                              loss_function=loss_function,
                              presentation_of_loss_value=100,
                              save_weights_path=self._weights_save_path,
                              save_info_path=self._info_save_path,
                              use_auto_graph=self._use_auto_graph)

        trainer.start()


if __name__ == "__main__":
    CoCaCommandTrainer()