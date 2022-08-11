from ospark.nn.cell.resnet_cell import ResnetCell
from ospark.nn.layers.convolution_layer import ConvolutionLayer
from typing import List, NoReturn, Optional
from ospark.backbone.backbone import Backbone
import tensorflow as tf


class ResnetBackbone(Backbone):

    def __init__(self,
                 obj_name: str,
                 cells: List[ResnetCell],
                 is_keep_blocks: bool,
                 is_training: Optional[bool]=None):
        super().__init__(obj_name=obj_name,
                         is_keep_blocks=is_keep_blocks,
                         is_training=is_training)
        self._cells = cells

    @property
    def cells(self) -> List[ResnetCell]:
        return self._cells

    def in_creating(self) -> NoReturn:
        self.assign(component=ConvolutionLayer.conv_bn_relu(obj_name="first_conv",
                                                            filter_size=[7, 7, 3, 64],
                                                            strides=[1, 2, 2, 1],
                                                            padding="SAME",
                                                            is_training=self.is_training))
        for cell in self.cells:
            self.assign(component=cell)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        output = self.assigned.first_conv.pipeline(input_data)
        output = tf.nn.max_pool2d(output, strides=2, ksize=3, padding="SAME")
        for cell in self.cells:
            output = cell.pipeline(output)
            if self.is_keep_blocks:
                self._block_register.append(output)
        output = self.block_register if self.is_keep_blocks else output
        return output