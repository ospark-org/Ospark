from ospark.trainer.exdeep_transformer import ExdeepTransformerTrainer
from ospark.data.generator.translate_data_generator import TranslateDataGenerator
from ospark.former.builder import build_exdeep_transformer
from ospark.predictor.translator import Translator
from typing import Optional
from sacrebleu.metrics import BLEU
import tensorflow_datasets as tfds
import numpy as np
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def text_encoder(folder_path: str,
                 vocab_file: str,
                 vocabulary_size: int,
                 datasets: Optional[list]=None) -> tfds.deprecated.text.SubwordTextEncoder:
    file = os.path.join(folder_path, vocab_file)
    try:
        subword_encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file(file)
        print(f"載入已建立的字典： {file}")
    except:
        print("沒有已建立的字典，從頭建立。")
        subword_encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            datasets,
            target_vocab_size=vocabulary_size)
        # 將字典檔案存下以方便下次 warm start
        subword_encoder.save_to_file(file)
    return subword_encoder


weight_path = "/Users/abnertsai/Documents/Ospark/ospark/samples"
file_name   = "exdeep_16_50_60_12_4_128.json"
# file_name     = "exdeep_32_50_6_6_8_512.json"


folder_path          = "/Users/abnertsai/Documents/Gitlab/self-attention/tensorflow_datasets"
# folder_path          = os.path.dirname(os.path.abspath(__file__))
dataset_name         = 'wmt14_translate/de-en'
train_vocab_file     = "de_vocab_file"
target_vocab_file    = "en_vocab_file"
vocabulary_size      = 40000



scale_rate = 4
model, batch_size, epoch_number, encoder_block_number, decoder_block_number, head_number, embedding_size = file_name.split(".")[0].split("_")

with open(os.path.join(weight_path, file_name), 'r') as fp:
    weights = json.load(fp)


ds = tfds.load(data_dir=folder_path,
               name=dataset_name,
               as_supervised=True)

test_data = ds["test"]



train_datasets, target_datasets = None, None
if not os.path.isfile(os.path.join(folder_path, train_vocab_file + ".subwords")):
    print("找不到 text encoder file，讀取 datasets 建立 text encoder")
    train_datasets, target_datasets = zip(*[[train_data.numpy(), target_data.numpy()]
                                            for train_data, target_data
                                            in test_data])


# 建立/讀取 text_encoder
train_data_text_encoder = text_encoder(folder_path=folder_path,
                                       vocab_file=train_vocab_file,
                                       vocabulary_size=vocabulary_size,
                                       datasets=train_datasets)

target_data_text_encoder = text_encoder(folder_path=folder_path,
                                        vocab_file=target_vocab_file,
                                        vocabulary_size=vocabulary_size,
                                        datasets=target_datasets)

exdeep_model = build_exdeep_transformer(encoder_block_number=int(encoder_block_number),
                                        decoder_block_number=int(decoder_block_number),
                                        head_number=int(head_number),
                                        embedding_size=int(embedding_size),
                                        scale_rate=scale_rate,
                                        class_number=target_data_text_encoder.vocab_size + 2,
                                        encoder_corpus_size=train_data_text_encoder.vocab_size + 2,
                                        decoder_corpus_size=target_data_text_encoder.vocab_size + 2)


# 建立 data_generator
data_generator = TranslateDataGenerator(datasets=test_data,
                                        train_data_encoder=train_data_text_encoder,
                                        target_data_encoder=target_data_text_encoder,
                                        batch_size=1,
                                        take_number=None,
                                        max_length=None)

predictor = Translator(model=exdeep_model,
                       input_text_encoder=train_data_text_encoder,
                       output_text_encoder=target_data_text_encoder,
                       predict_max_length=100)

predictor.restore_weights(weights=weights)

bleu = BLEU()

scores = []
for before, reference in test_data.batch(1):
    input_data = before[0].numpy().decode()
    reference  = reference[0].numpy().decode()
    prediction = predictor.predict(input_data=input_data)
    score = bleu.corpus_score(prediction, reference)
    scores.append(score.score)

print(np.mean(np.array(scores)))

