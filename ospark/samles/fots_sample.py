from ospark.data.generator.fots_data_generator import FOTSDataGenerator
from ospark.ocr.fots import FOTSTrainer
import tensorflow as tf
import json
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
data_generator = FOTSDataGenerator(training_data_path=training_data_folder,
                                   target_data_path=target_data_folder,
                                   training_file_name=training_list,
                                   target_file_name=target_list,
                                   batch_size=4,
                                   filter_height=2,
                                   filter_words="###",
                                   image_size=[1280, 720],
                                   image_shrunk=0.25)

# create fots trainer
trainer = FOTSTrainer.create_attention_fots(data_generator=data_generator,
                                            corpus=corpus["char_to_index"],
                                            batch_size=4,
                                            epoch_number=30,
                                            optimizer=tf.keras.optimizers.Adam,
                                            learning_rate=0.001)

# start training
trainer.start()
