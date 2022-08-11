from ospark.nn.optimizer.learning_rate_schedule import TransformerWarmup
from ospark.models.builder import FormerBuilder
from ospark.data.generator.translate_data_generator import TranslateDataGenerator
from ospark.trainer.exdeep_transformer import ExdeepTransformerTrainer
from ospark.trainer.transformer_trainer import TransformerTrainer
from ospark.nn.loss_function import SparseCategoricalCrossEntropy
from ospark.data.encoder import LanguageDataEncoder
from typing import Optional
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import tensorflow as tf
import json
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


folder_path          = "/Users/abnertsai/Documents/Gitlab/self-attention/tensorflow_datasets" #
# folder_path          = os.path.dirname(os.path.abspath(__file__))

# dataset_name         = 'wmt14_translate/de-en'
# train_vocab_file     = "de_vocab_file"
# target_vocab_file    = "en_vocab_file"
# train_vocab_file     = "en_vocab_file"
# target_vocab_file    = "de_vocab_file"


dataset_name         = 'ted_hrlr_translate/pt_to_en'
train_vocab_file     = "pt_vocab_file"
target_vocab_file    = "en_vocab_file"

batch_size           = 8
max_length           = None
take_number          = 1000
encoder_block_number = 4
decoder_block_number = 4
head_number          = 4
embedding_size       = 128
scale_rate           = 4
dropout_rate         = 0.0
use_graph_mode       = False
epoch_number         = 50
save_times           = 5
vocabulary_size      = 32000
use_profiling_phase  = False
save_init_weights    = False
use_restore          = True


if use_profiling_phase:
    model_name = "exdeep"
else:
    model_name = "transformer"

# 權重儲存路徑
save_path = os.path.join(folder_path,
            f"{model_name}_{batch_size}_{epoch_number}_{encoder_block_number}_{decoder_block_number}_{head_number}_{embedding_size}.json")

init_weights_path = os.path.join(folder_path,
            f"{encoder_block_number}_{decoder_block_number}_{head_number}_{embedding_size}.json")

# 讀取訓練用的 datasets wmt14
print("讀取訓練用資料")
ds = tfds.load(data_dir=folder_path,
               name=dataset_name,
               as_supervised=True)

train_examples, val_examples = ds["train"], ds["validation"]
print(type(train_examples))
train_datasets, target_datasets = None, None
if not os.path.isfile(os.path.join(folder_path, train_vocab_file + ".subwords")):
    print("找不到 text encoder file，讀取 datasets 建立 text encoder")
    train_datasets, target_datasets = zip(*[[train_data.numpy(), target_data.numpy()]
                                            for train_data, target_data
                                            in train_examples])

# 建立/讀取 text_encoder
train_data_text_encoder = text_encoder(folder_path=folder_path,
                                       vocab_file=train_vocab_file,
                                       vocabulary_size=vocabulary_size,
                                       datasets=train_datasets)

target_data_text_encoder = text_encoder(folder_path=folder_path,
                                        vocab_file=target_vocab_file,
                                        vocabulary_size=vocabulary_size,
                                        datasets=target_datasets)

print("拆分 train data")
training_data, target_data = list(zip(*train_examples))
# 建立 data_generator
print("建立 data generator")
data_encoder   = LanguageDataEncoder(train_data_encoder=train_data_text_encoder,
                                     label_data_encoder=target_data_text_encoder)

data_generator = TranslateDataGenerator(training_data=training_data,
                                        target_data=target_data,
                                        data_encoder=data_encoder,
                                        batch_size=batch_size,
                                        max_length=max_length,
                                        max_token=3000)

if os.path.isfile(init_weights_path) and use_restore:
    print("讀取 init weights")
    with open(init_weights_path, 'r') as fp:
        weights = json.load(fp)
else:
    print("Use random weights")
    weights = None

print("建立模型")
exdeep_model = FormerBuilder.exdeep_transformer(encoder_block_number=encoder_block_number,
                                                decoder_block_number=decoder_block_number,
                                                trained_weights=weights,
                                                head_number=head_number,
                                                embedding_size=embedding_size,
                                                scale_rate=scale_rate,
                                                class_number=target_data_text_encoder.vocab_size + 2,
                                                encoder_corpus_size=train_data_text_encoder.vocab_size + 2,
                                                decoder_corpus_size=target_data_text_encoder.vocab_size + 2,
                                                is_training=True,
                                                dropout_rate=dropout_rate)

# performer_model = FormerBuilder.performer(block_number=encoder_block_number,
#                                           head_number=head_number,
#                                           embedding_size=embedding_size,
#                                           scale_rate=scale_rate,
#                                           random_projections_number=16,
#                                           class_number=target_data_text_encoder.vocab_size + 2,
#                                           encoder_corpus_size=train_data_text_encoder.vocab_size + 2,
#                                           decoder_corpus_size=target_data_text_encoder.vocab_size + 2,
#                                           is_training=True,
#                                           dropout_rate=dropout_rate)

# 設定 learning_rate、optimizer、loss_function
learning_rate = TransformerWarmup(model_dimension=embedding_size, warmup_step=4000.)
# optimizer     = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
optimizer     = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98)
loss_function = SparseCategoricalCrossEntropy()


print("建立 trainer")
trainer = ExdeepTransformerTrainer(data_generator=data_generator,
                                   model=exdeep_model,
                                   epoch_number=epoch_number,
                                   optimizer=optimizer,
                                   loss_function=loss_function,
                                   save_times=save_times,
                                   save_path=save_path,
                                   use_profiling_phase=use_profiling_phase,
                                   use_auto_graph=use_graph_mode,
                                   save_init_weights=save_init_weights,
                                   init_weights_path=init_weights_path)

# trainer = TransformerTrainer(data_generator=data_generator,
#                              model=performer_model,
#                              epoch_number=epoch_number,
#                              optimizer=optimizer,
#                              loss_function=loss_function,
#                              save_times=save_times,
#                              save_path=save_path,
#                              use_auto_graph=use_graph_mode,
#                              save_init_weights=save_init_weights,
#                              init_weights_path=init_weights_path)


print("開始訓練")
trainer.start()