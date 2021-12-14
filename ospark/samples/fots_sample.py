from ospark.data.generator.fots_data_generator import FOTSDataGenerator
from ospark.ocr.fots import FOTSTrainer
from ospark.nn.optimizer.learning_rate_schedule import TransformerWarmup
import tensorflow as tf
import json
import os

os_envi = "mac"

if os_envi == "mac":
    # dataset path
    folder_path          = "/Users/abnertsai/Documents/ICDAR/ch4/"
    training_data_folder = folder_path + "training_data"
    target_data_folder   = folder_path + "ch4_training_localization_transcription_gt"
    corpus_path          = folder_path + "corpus.json"

elif os_envi == "gpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # dataset path
    folder_path          = "/home/ai/abner/ICDAR/ICDAR/ch4/"
    training_data_folder = folder_path + "training_data"
    target_data_folder   = folder_path + "ch4_training_localization_transcription_gt"
    corpus_path          = folder_path + "corpus.json"

elif os_envi == "maki_gpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # dataset path
    folder_path          = os.path.dirname(os.path.abspath(__file__))
    training_data_folder = folder_path + "/dataset/ICDAR/training_data"
    target_data_folder   = folder_path + "/dataset/ICDAR/ch4_training_localization_transcription_gt"
    corpus_path          = folder_path + "/dataset/ICDAR/corpus.json"

# load corpus
with open(corpus_path, 'r') as fp:
    corpus = json.load(fp)

# get data files
training_list = os.listdir(training_data_folder)
target_list   = os.listdir(target_data_folder)

# create data generator
data_generator = FOTSDataGenerator(training_data_folder=training_data_folder,
                                   target_data_folder=target_data_folder,
                                   training_files_name=training_list,
                                   target_files_name=target_list,
                                   batch_size=2,
                                   filter_height=8,
                                   filter_words={"###"},
                                   target_size=[1280, 720],
                                   image_shrunk_scale=0.25)

# create fots trainer
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fots_weight.json")
trainer = FOTSTrainer.create_attention_fots(data_generator=data_generator,
                                            corpus=corpus["char_to_index"],
                                            batch_size=16,
                                            epoch_number=50,
                                            optimizer=tf.keras.optimizers.Adam(learning_rate=TransformerWarmup(model_dimension=512)),
                                            reg_optimizer=tf.keras.optimizers.Adam(learning_rate=TransformerWarmup(model_dimension=512)),
                                            save_path=save_path
                                            )

if os.path.isfile(save_path):
    with open(save_path, 'r') as fp:
        weight = json.load(fp)
    trainer.restore_weights(weights=weight)

# start training
trainer.start()
