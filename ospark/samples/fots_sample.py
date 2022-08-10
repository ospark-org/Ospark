from ospark.data.generator.fots_data_generator import FOTSDataGenerator
from ospark.ocr.fots import FOTSTrainer
from ospark.nn.optimizer.learning_rate_schedule import TransformerWarmup
from ospark.data.folder import DataFolder
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

# get data folder
data_folder   = DataFolder(train_data_folder=training_data_folder, label_data_folder=target_data_folder)
# create data generator
data_generator = FOTSDataGenerator(data_folder=data_folder,
                                   batch_size=2,
                                   height_threshold=8,
                                   ignore_words={"###"},
                                   target_size=[1280, 720],
                                   image_shrink_scale=0.25)

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fots_weight.json")
if os.path.isfile(save_path):
    with open(save_path, 'r') as fp:
        weight = json.load(fp)
else:
    weight = None

# create fots trainer
trainer = FOTSTrainer.create_attention_fots(data_generator=data_generator,
                                            corpus=corpus["char_to_index"],
                                            retrained_weights=weight,
                                            batch_size=16,
                                            epoch_number=50,
                                            optimizer=tf.keras.optimizers.Adam(learning_rate=TransformerWarmup(model_dimension=512)),
                                            reg_optimizer=tf.keras.optimizers.Adam(learning_rate=TransformerWarmup(model_dimension=512)),
                                            save_path=save_path
                                            )

# start training
trainer.start()
