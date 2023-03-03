import ospark
from ospark.nn.layers import Layer
from typing import Optional
import tensorflow as tf


class ConvolutionAttention(Layer):

    def __init__(self,
                 obj_name: str,
                 mlp_layer: Layer,
                 filter_size: Optional[list]=None,
                 is_training: Optional[bool]=None,
                 training_phase: Optional[bool]=None):
        super(ConvolutionAttention, self).__init__(obj_name=obj_name,
                                                   is_training=is_training)
        self._mlp_layer   = mlp_layer
        self._filter_size = filter_size or [7, 7]
        self._filter      = ospark.weight_initializer.glorot_uniform(obj_name="spatial_filter",
                                                                     shape=self._filter_size + [2, 1],
                                                                     trainable=is_training)

    @property
    def mlp_layer(self) -> Layer:
        return self._mlp_layer

    @property
    def filter_size(self) -> list:
        return self._filter_size

    @property
    def filter(self) -> ospark.weight:
        return self._filter

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        channel_attention_result = self.channel_attention(input_data=input_data)
        attention_result         = tf.multiply(input_data, channel_attention_result)
        spatial_attention_result = self.spatial_attention(input_data=input_data)
        attention_result         = tf.multiply(attention_result, spatial_attention_result)
        return attention_result

    def channel_attention(self, input_data: tf.Tensor):
        avg_pool   = tf.nn.avg_pool2d(input=input_data, ksize=input_data.shape[1:3], strides=[1, 1, 1, 1]) # [B,1,1,C]
        max_pool   = tf.nn.max_pool2d(input=input_data, ksize=input_data.shape[1:3], strides=[1, 1, 1, 1]) # [B,1,1,C]
        pools      = tf.squeeze(tf.concat([avg_pool, max_pool], axis=0)) # [2 * B, C]
        mlp_output = self._mlp_layer.pipeline(input_data=pools)
        avg_pool, max_pool = tf.split(mlp_output, num_or_size_splits=2, axis=0)
        output     = tf.nn.sigmoid(avg_pool + max_pool)
        return output[:, tf.newaxis, tf.newaxis, :]

    def spatial_attention(self, input_data: tf.Tensor):
        avg_pool    = tf.nn.avg_pool2d(input=input_data, ksize=[1, 1, 1, input_data.shape[-1]], strides=[1, 1, 1, 1]) # [B,H,W,1]
        max_pool    = tf.nn.max_pool2d(input=input_data, ksize=[1, 1, 1, input_data.shape[-1]], strides=[1, 1, 1, 1]) # [B,H,W,1]
        pools       = tf.concat([avg_pool, max_pool], axis=-1) # [B, H, W, 2]
        conv_output = tf.nn.conv2d(input=pools, filters=self.filter, strides=[1, 1, 1, 1], padding="SAME") # [B, H, W, 1]
        output      = tf.nn.sigmoid(conv_output)
        return output

if __name__ == "__main__":
    from ospark.nn.layers.dense_layer import DenseLayer
    mlp_layer = DenseLayer(obj_name="aa_layer", input_dimension=32, hidden_dimension=[16, 32])
    input_data = tf.random.normal(shape=[4,128,128,32])
    cbam = ConvolutionAttention("CBAM", mlp_layer)
    cbam.create()
    print(cbam.pipeline(input_data=input_data))