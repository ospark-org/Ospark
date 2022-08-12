import ospark.utility.weight_initializer
from ospark.nn.layers import *
import ospark


class EmbeddingLayer(Layer):

    def __init__(self,
                 obj_name: str,
                 embedding_dimension: int,
                 corpus_size: int,
                 is_training: Optional[bool]=None):
        super().__init__(obj_name=obj_name, is_training=is_training)
        self._embedding_dimension = embedding_dimension
        self._corpus_size         = corpus_size

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @property
    def corpus_size(self) -> int:
        return self._corpus_size

    def in_creating(self) -> NoReturn:
        with tf.device("cpu:0"):
            self._embedding_layer = ospark.utility.weight_initializer.uniform(obj_name="embedding_layer",
                                                                              shape=[self.corpus_size,
                                                                 self.embedding_dimension])
            # self.assign(component=ospark.weight.uniform(obj_name="embedding_layer",
            #                                             weight_shape=[self.corpus_size,
            #                                                           self.embedding_dimension]))

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        with tf.device("cpu:0"):
            sequence = tf.nn.embedding_lookup(self._embedding_layer, ids=input_data)
        return sequence
        # mask = tf.cast(tf.math.not_equal(input_data, 0), tf.float32)[:, :, tf.newaxis]
        # input_data = tf.one_hot(indices=input_data, depth=self.corpus_size) * mask
        # return tf.matmul(input_data, self.assigned.embedding_layer)