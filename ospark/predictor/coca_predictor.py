import numpy as np
import tensorflow as tf
from ospark.models.coca import CoCa
from ospark.nn.model import Model


class CoCaPredictor:

    def __init__(self,
                 model: CoCa):
        self._model  = model

    @property
    def model(self) -> CoCa:
        return self._model

    def predict(self, image: np.ndarray):
        image  = tf.convert_to_tensor(image, dtype=tf.float32)
        result = self.model.pipeline(images=image)
        return result

if __name__ == "__main__":
    from PIL import Image
    import json
    from ospark.models.BEiT import BEiT
    from ospark.nn.layers.embedding_layer import EmbeddingLayer

    training_data_path = "/Users/donggicai1991/Documents/Ospark/trainin_dataset.ndjson"
    corpus_path = "/Users/donggicai1991/Documents/Ospark/corpus.json"

    with open(corpus_path, 'r') as fp:
        corpus = json.load(fp)

    with open("/Users/donggicai1991/Documents/airflow/weights/coca/coca_weights_1.json", 'r') as fp:
        coca_weights = json.load(fp)

    # with open("~/Documents/airflow/weights/coca/coca_info.json", 'r') as fp:
    #     coca_info = json.load(fp)

    image_size = [256, 256]
    patch_size = [8, 8]
    block_number = 6
    head_number = 8
    embedding_size = 256
    scale_rate = 4
    corpus_size = len(corpus)
    dropout_rate = 0.1

    image_encoder = BEiT(obj_name="image_encoder",
                         image_size=image_size,
                         patch_size=patch_size,
                         block_number=block_number,
                         head_number=head_number,
                         embedding_size=embedding_size,
                         scale_rate=scale_rate,
                         dropout_rate=dropout_rate,
                         training_phase=False,
                         is_training=False)

    embedding_layer = EmbeddingLayer(obj_name="embedding_layer",
                                     embedding_dimension=embedding_size,
                                     corpus_size=corpus_size)

    model = CoCa(obj_name="coca",
                 image_encoder=image_encoder,
                 head_number=head_number,
                 embedding_size=embedding_size,
                 embedding_layer=embedding_layer,
                 corpus=corpus,
                 scale_rate=scale_rate,
                 dropout_rate=dropout_rate,
                 trained_weights=coca_weights,
                 is_training=False,
                 training_phase=False,
                 use_predict_result=True)

    # model = Model.create_from_info(model_info=coca_info, trained_weights=coca_weights)

    predictor = CoCaPredictor(model=model)
    import pathlib
    # images_path = pathlib.Path("/Volumes/T7/23.02.20/標註/D_手開三聯式/完成/img(DA)")
    images_path = pathlib.Path("/Volumes/File Server/01.TA總所/26.AI/42651975_意德/112/5-6")
    # images_path = pathlib.Path("/Volumes/T7/112_3-4_status/imgs")
    files       = [path for path in images_path.iterdir() if str(path).split("/")[-1][-3:] == "jpg" and str(path).split("/")[-1][0] != "."]

    for file_path in files:
        image = np.array(Image.open(file_path).resize(size=[256,256]))[np.newaxis, ...]
        print(predictor.predict(image=image))
        raise


