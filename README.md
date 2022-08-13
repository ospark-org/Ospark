Ospark currently based on Tensorflow 2.3+ and Python 3.7+ to easy and quick build/tune "former series models" that include Transformer, Informer, ExDeep Performance and FOTS-attention these powerful models. Osaprk wishes to support AI algorithm engineers, researchers and any deep learning learners to code and math balance. Now is v0.0.4 beta release.

### Quick Install

```bash
$ pip install ospark
```

### Quick Usage
```python
from ospark import weight_initializer
weight = weight_initializer.normal(obj_name=weight_name,
                                   shape=weight_shape,
                                   trainable=True)
```

Some common weight initial methods as below, and the argument "trainable" default to True:

- `truncated_normal(obj_name, shape, trainable)`
    
    **obj_name: str**
    
    **shape: List[int]**
    
    **trainable: Optional[bool]**
    

- `normal(obj_name, shape, trainable)`
    
    **obj_name: str**
    
    **shape: List[int]**
    
    **trainable: Optional[bool]**
   
 
- `uniform(obj_name, shape, trainable)`
    
    **obj_name: str**
    
    **shape: List[int]**
    
    **trainable: Optional[bool]**
    

- `ones(obj_name, shape, trainable)`
    
    **obj_name: str**
    
    **shape: List[int]**
    
    **trainable: Optional[bool]**
    

- `zeros(obj_name, shape, trainable)`
    
    **obj_name: str**
    
    **shape: List[int]**
    
    **trainable: Optional[bool]**
    

- `glorot_uniform(obj_name, shape, trainable)`
    
    **obj_name: str**
    
    **shape: List[int]**
    
    **trainable: Optional[bool]**
    

### Build Layers

```python
from ospark import Layer, weight_initializer
import tensorflow as tf

class FullyConnectedLayer(Layer):

    def __init__(self, obj_name: str, weight_shape: list, is_training: bool):
	    super().__init__(obj_name=obj_name, is_training=is_training)
	    self._weight_shape = weight_shape

    @property
    def weight_shape(self) -> list:
        return self._weight_shape

    def in_creating(self):
        self._weight = weight_initializer.glorot_uniform(obj_name="weight", shape=[128,128], trainable=True)
        self._bias   = weight_initializer.zeros(obj_name="bias",shape=[128],trainable=True)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        output = tf.matmul(input_data, self._weight) + self._bias
        return output

```

### Build Blocks

This demonstration shows how to possibly making easy way to experiment any blocks by your own ideas.

```python
from ospark import Block
import tensorflow as tf
from typing import Optional

# the Ospark default support blocks are:
from ospark.nn.block.resnet_block import Block1, Block2 # the Block1: [1X1, 3X3, 1X1], Block2: [3X3, 3X3]
from ospark.nn.block.transformer_block import transformer_decoder_block, transformer_encoder_block
from ospark.nn.block.vgg_block import VGGBlock

# if you need Dense-block, makes it:
class DenseBlock(Block):

    def __init__(self, 
                 obj_name: str,
                 input_shape: int,
                 output_shape: int,
                 hidden_dimension: int,
                 is_training: bool):
        super().__init__(obj_name=obj_name, is_training=is_training)
        self._layer_1 = FullyConnectedLayer("layer_1", [input_shape, hidden_dimension], is_training)
        self._layer_2 = FullyConnectedLayer("layer_2", [hidden_dimension, output_shape], is_training)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        layer_output = self._layer_1.pipeline(input_data)
        layer_output = self._layer_2.pipeline(layer_output)
        return layer_output

# if you need to create like Transformer encode/decode blocks, to extra import 2 modeules then doing something as below demo code:
from ospark.nn.layers.self_attention import SelfAttentionLayer
from ospark.nn.layers.feed_forward import FeedForwardLayer

# demo code of Transformer Block
class TransformerEncoderBlock(Block):

    def __init__(self,
                 obj_name: str,
                 embedding_size: int,
                 head_number: int,
                 dropout_rate: float,
                 scale_rate: int,
                 is_training: bool):
        super().__init__(obj_name=obj_name, is_training=is_training)
        self._attention   = SelfAttentionLayer(obj_name="attention_layer", 
                                               embedding_size=embedding_size,
                                               head_number=head_number,
                                               dropout_rate=dropout_rate,
                                               is_training=is_training,
                                               use_look_ahead=False)
        self._feedforward = FeedForwardLayer(obj_name="attention_layer", 
                                             embedding_size=embedding_size,
                                             scale_rate=scale_rate,
                                             dropout_rate=dropout_rate,
                                             is_training=is_training)
    
    @property
    def attention(self) -> SelfAttentionLayer:
        return self._attention

    @property
    def feedforward(self) -> FeedForwardLayer:
        return self._feedforward
    
    def pipeline(self, input_data: tf.Tensor, mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        output = self.attention.pipeline(input_data=input_data, mask=mask)
        output = self.feedforward.pipeline(input_data=output)
        return output
	
```

### Build Model

```python
from ospark import Model

class ClassifyModel(Model):

    def __init__(self, 
                 obj_name: str, 
                 output_dimension: int,
                 is_training: bool,
                 trained_weights: dict={}):
        super().__init__(obj_name=obj_name, trained_weights=trained_weights, is_training=is_training)
        self._block_1 = DenseBlock("block_1", input_shape=128, output_shape=64, hidden_dimension=256, is_training=is_training)
        self._block_2 = DenseBlock("block_2", input_shape=256, output_shape=128, hidden_dimension=512, is_training=is_training)
    
        self._classify_layer = FullyConnectedLayer("classify_layer", [512, output_dimension], is_training=is_training)
        
    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        block_output = self._block_1.pipeline(input_data)
        block_output = self._block_2.pipeline(block_output)
        model_output = self._classfy_layer.pipeline(block_output)
        return model_output
```

### Fetch Weights

```python
model = ClassifyModel("classify_model", 3, True)
weights = model.get_weights()
print("weights: ", weights.keys())
# saving trained weights to any folder:
# ...
```
```bash
weights:  dict_keys(['classify_model/block_2/layer_2/weight', 'classify_model/block_2/layer_2/bias', 'classify_model/block_2/layer_1/weight', 'classify_model/block_2/layer_1/bias', 'classify_model/block_1/layer_2/weight', 'classify_model/block_1/layer_2/bias', 'classify_model/block_1/layer_1/weight', 'classify_model/block_1/layer_1/bias', 'classify_model/classify_layer/weight', 'classify_model/classify_layer/bias'])
```

### Restore Model

```python
# Ospark supports default loader:
from ospark.data.data_operator import JsonOperator as jo
old_weights = jo.load(path="weights_saved_path")

# or you can load the file by your method:
# old_weights = load_weights(path="weights_saved_path")

model = ClassifyModel("classify_model", 3, True, trained_weights=old_weights)
```


