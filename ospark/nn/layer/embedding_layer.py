from ospark.nn.layer import *
import ospark


class EmbeddingLayer(Layer):

    def __init__(self,
                 obj_name: str,
                 embedding_dimension: int,
                 corpus_size: int):
        super().__init__(obj_name=obj_name)
        self._embedding_dimension = embedding_dimension
        self._corpus_size         = corpus_size

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @property
    def corpus_size(self) -> int:
        return self._corpus_size

    def on_creating(self) -> NoReturn:
        with tf.device("cpu:0"):
            self.assign(component=ospark.weight.uniform(obj_name="embedding_layer",
                                                        weight_shape=[self.corpus_size,
                                                                      self.embedding_dimension]))

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        with tf.device("cpu:0"):
            sequence = tf.nn.embedding_lookup(self.assigned.embedding_layer, ids=input_data)
        return sequence
        # mask = tf.cast(tf.math.not_equal(input_data, 0), tf.float32)[:, :, tf.newaxis]
        # input_data = tf.one_hot(indices=input_data, depth=self.corpus_size) * mask
        # return tf.matmul(input_data, self.assigned.embedding_layer)