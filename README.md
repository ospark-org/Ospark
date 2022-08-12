# Ospark
Ospark currently based on Tensorflow 2.3+ and Python 3.7+ to easy and quick build/tune "former series models" that include Transformer, Informer, ExDeep Performance and FOTS-attention these powerful models by yourself. Now is alpha version we're internal testing and will release the beta version as soon.

# Ospark Guidelines

### ***Currently in beta version.***

# Install

```bash
$ pip install ospark
```

# Build yourself model.

Ospark divides the model into Layer, Block and Model. 

## Weight initializer

Osaprk provides some simple weight initialization methods.

First, import weight_initializer

```python
from ospark import weight_initializer
```

initial weight:

```python
weight = weight_initializer.normal(obj_name=weight_name,
                                   shape=weight_shape,
                                   trainable=True)
```

The initialization methods implemented in weight_initializer are as follows:

Note: trainable default is True.

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
    

## Build Layer

***Note: Weight-related processing must be placed in in_creating***

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

## Build block

```python
from ospark import Block

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
```

## Build model

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

## Get weights:

```python
model = ClassifyModel("classify_model", 3, True)
weights = model.get_weights()
print("weights: ",weights.kesy())
```

## Result
```bash
Initialize weight classify_model/block_2/layer_2/weight.
Initialize weight classify_model/block_2/layer_2/bias.
Initialize weight classify_model/block_2/layer_1/weight.
Initialize weight classify_model/block_2/layer_1/bias.
Initialize weight classify_model/block_1/layer_2/weight.
Initialize weight classify_model/block_1/layer_2/bias.
Initialize weight classify_model/block_1/layer_1/weight.
Initialize weight classify_model/block_1/layer_1/bias.
Initialize weight classify_model/classify_layer/weight.
Initialize weight classify_model/classify_layer/bias.
weights:  dict_keys(['classify_model/block_2/layer_2/weight', 'classify_model/block_2/layer_2/bias', 'classify_model/block_2/layer_1/weight', 'classify_model/block_2/layer_1/bias', 'classify_model/block_1/layer_2/weight', 'classify_model/block_1/layer_2/bias', 'classify_model/block_1/layer_1/weight', 'classify_model/block_1/layer_1/bias', 'classify_model/classify_layer/weight', 'classify_model/classify_layer/bias'])
```

## Restore model:

```python
old_weights = load_weights(path="weights_save_path")
model = ClassifyModel("classify_model", 3, True, trained_weights=old_weights)
```
