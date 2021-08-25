import ospark
from ospark.nn.cell.resnet_cell import ResnetCell
from ospark.nn.component.basic_module import BasicModule
from ospark.nn.layer.convolution_layer import ConvolutionLayer
from ospark.nn.component.weight import Weight
from typing import List, NoReturn, Optional
import tensorflow as tf

class Resnet(BasicModule):

    def __init__(self,
                 obj_name: str,
                 cells: List[ResnetCell],
                 first_conv_size: List[int],
                 first_conv_strides: int,
                 pooling_strides: int,
                 pooling_size: int,
                 trainable: Optional[bool]=True,
                 classify_layer: Optional[Weight]=None
                 ):
        super().__init__(obj_name=obj_name)
        self._cells              = cells
        self._classify_layer     = classify_layer
        self._first_conv_size    = first_conv_size
        self._first_conv_strides = first_conv_strides
        self._pooling_strides    = pooling_strides
        self._pooling_size       = pooling_size
        self._trainable          = trainable

    @property
    def cells(self) -> List[ResnetCell]:
        return self._cells

    @property
    def classify_layer(self) -> Optional[Weight]:
        return self._classify_layer

    @property
    def first_conv_size(self) -> List[int]:
        return self._first_conv_size

    @property
    def first_conv_strides(self) -> int:
        return self._first_conv_strides

    @property
    def pooling_size(self) -> int:
        return self._pooling_size

    @property
    def pooling_strides(self) -> List[int]:
        return [1, self._pooling_strides, self._pooling_strides, 1]

    @property
    def trainable(self) -> bool:
        return self._trainable

    def initialize(self) -> NoReturn:
        for cell in self.cells:
            self.assign(component=cell)
        self.assign(component=ConvolutionLayer.bn_relu_conv(obj_name="first_conv",
                                                            filter_size=self.first_conv_size,
                                                            strides=[1, self.first_conv_strides, self.first_conv_strides, 1],
                                                            padding="SAME",
                                                            trainable=self.trainable))
        if self.classify_layer is not None:
            self.assign(component=self.classify_layer, name="classify_layer")

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        output = self.assigned.first_conv(input_data)
        output = tf.nn.max_pool2d(output, strides=self.pooling_strides, ksize=self.pooling_size)
        for cell in self.cells:
            output = cell(output)
        if self.classify_layer is not None:
            output = tf.reduce_mean(output , axis=[1, 2])
            output = tf.matmul(output, self.assigned.classify_layer)
        return output

    def create(self) -> NoReturn:
        super().create("model")
