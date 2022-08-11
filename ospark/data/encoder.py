from tensorflow_datasets.core.deprecated.text.subword_text_encoder import SubwordTextEncoder


class LanguageDataEncoder:

    def __init__(self, train_data_encoder: SubwordTextEncoder, label_data_encoder: SubwordTextEncoder):
        self._train_data_encoder = train_data_encoder
        self._label_data_encoder = label_data_encoder

    @property
    def train_data_encoder(self) -> SubwordTextEncoder:
        return self._train_data_encoder

    @property
    def label_data_encoder(self) -> SubwordTextEncoder:
        return self._label_data_encoder

    def encode_train_data(self, input_data: list) -> list:
        return self.train_data_encoder.encode(input_data)

    def encode_label_data(self, input_data: list) -> list:
        return self.label_data_encoder.encode(input_data)