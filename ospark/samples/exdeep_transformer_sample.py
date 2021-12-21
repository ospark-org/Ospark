from ospark.nn.optimizer.learning_rate_schedule import TransformerWarmup
from ospark.former.builder import build_exdeep_transformer
from ospark.data.generator.translate_data_generator import TranslateDataGenerator
from ospark.trainer.exdeep_transformer import ExdeepTransformerTrainer
from ospark.test_loss import Transformer
from ospark.nn.loss_function import SparseCategoricalCrossEntropy
from typing import Optional
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import tensorflow as tf
import os
import time
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
dataset_name         = 'wmt14_translate/de-en'
# train_vocab_file     = "de_vocab_file"
# target_vocab_file    = "en_vocab_file"
train_vocab_file     = "en_vocab_file"
target_vocab_file    = "de_vocab_file"
batch_size           = 16
max_length           = 100
take_number          = 30000
encoder_block_number = 60
decoder_block_number = 12
head_number          = 4
embedding_size       = 128
scale_rate           = 4
dropout_rate         = 0.3
use_graph_mode       = True
epoch_number         = 50
save_times           = 5
vocabulary_size      = 2**13
use_profiling_phase  = True
save_init_weights    = True
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
ds = tfds.load(data_dir=folder_path,
               name=dataset_name,
               as_supervised=True)

train_examples, val_examples = ds["train"], ds["validation"]

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


# 建立 data_generator
data_generator = TranslateDataGenerator(datasets=train_examples,
                                        train_data_encoder=train_data_text_encoder,
                                        target_data_encoder=target_data_text_encoder,
                                        batch_size=batch_size,
                                        take_number=take_number,
                                        max_length=max_length)

# 建立模型
exdeep_model = build_exdeep_transformer(encoder_block_number=encoder_block_number,
                                        decoder_block_number=decoder_block_number,
                                        head_number=head_number,
                                        embedding_size=embedding_size,
                                        scale_rate=scale_rate,
                                        class_number=target_data_text_encoder.vocab_size + 2,
                                        encoder_corpus_size=train_data_text_encoder.vocab_size + 2,
                                        decoder_corpus_size=target_data_text_encoder.vocab_size + 2,
                                        is_training=True,
                                        dropout_rate=dropout_rate)

# 設定 learning_rate、optimizer、loss_function
learning_rate = TransformerWarmup(model_dimension=embedding_size, warmup_step=4000.)
# optimizer     = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
optimizer     = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98)
loss_function = SparseCategoricalCrossEntropy()


# print("建立 trainer")
# trainer = ExdeepTransformerTrainer(data_generator=data_generator,
#                                    model=exdeep_model,
#                                    epoch_number=epoch_number,
#                                    optimizer=optimizer,
#                                    loss_function=loss_function,
#                                    save_times=save_times,
#                                    save_path=save_path,
#                                    use_profiling_phase=use_profiling_phase,
#                                    use_auto_graph=use_graph_mode,
#                                    save_init_weights=save_init_weights,
#                                    init_weights_path=init_weights_path)
#
# if os.path.isfile(init_weights_path) and use_restore:
#     print("讀取 init weights")
#     with open(init_weights_path, 'r') as fp:
#         weights = json.load(fp)
#     trainer.restore_weights(weights=weights)
# else:
#     print("Use random weights")
#
# print("開始訓練")
# trainer.start()


# 建立 tensorflow transformer
transformer = Transformer(num_layers=6,
                          d_model=512,
                          num_heads=4,
                          dff=512,
                          input_vocab_size=train_data_text_encoder.vocab_size + 2,
                          target_vocab_size=target_data_text_encoder.vocab_size + 2,
                          pe_input=2000,
                          pe_target=2000,
                          rate=0.0)

train_loss     = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

# 計算訓練準確率
def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


# 串接訓練流程
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  with tf.GradientTape() as tape:
    predictions, _ = transformer([inp, tar_inp],
                                 training = True)
    loss = loss_function(predictions, tar_real)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(accuracy_function(tar_real, predictions))

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                         optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')


# 開始訓練
for epoch in range(epoch_number):
  start = time.time()
  print("training start.")

  train_loss.reset_states()
  train_accuracy.reset_states()

  # inp -> portuguese, tar -> english
  for (batch, (inp, tar)) in enumerate(data_generator):
    train_step(inp, tar)

    if batch % 1 == 0:
      print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')