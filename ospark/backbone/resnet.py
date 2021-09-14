from ospark.nn.cell.resnet_cell import ResnetCell
from ospark.nn.layer.convolution_layer import ConvolutionLayer
from typing import List, NoReturn, Optional
from ospark.nn.cell.resnet_cell import resnet_cells
from ospark.backbone.backbone import Backbone
import tensorflow as tf


class ResnetBackbone(Backbone):

    def __init__(self,
                 obj_name: str,
                 cells: List[ResnetCell],
                 use_catch: bool,
                 trainable: Optional[bool]=True):
        super().__init__(obj_name=obj_name,
                         use_catch=use_catch,
                         trainable=trainable)
        self._cells = cells

    @property
    def cells(self) -> List[ResnetCell]:
        return self._cells

    def on_creating(self) -> NoReturn:
        self.assign(component=ConvolutionLayer.conv_bn_relu(obj_name="first_conv",
                                                            filter_size=[7, 7, 3, 64],
                                                            strides=[1, 2, 2, 1],
                                                            padding="SAME",
                                                            trainable=self.trainable))
        for cell in self.cells:
            self.assign(component=cell)

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        output = self.assigned.first_conv(input_data)
        output = tf.nn.max_pool2d(output, strides=2, ksize=3, padding="SAME")
        for cell in self.cells:
            output = cell(output)
            if self.use_catch:
                self._catch_box.append(output)
        output = self.catch_box if self.use_catch else output
        return output


def resnet_50(trainable: bool, catch_output: bool) -> ResnetBackbone:
    cells_number   = [3, 4, 6, 3]
    input_channels = [64, 256, 512, 1024]
    main_channels  = [64, 128, 256, 512]
    scale_rate     = 4
    cells          = resnet_cells(cells_number=cells_number,
                                  input_channels=input_channels,
                                  main_channels=main_channels,
                                  scale_rate=scale_rate,
                                  block_type="block_1",
                                  trainable=trainable)
    backbone = ResnetBackbone(obj_name="resnet_50",
                              cells=cells,
                              use_catch=catch_output,
                              trainable=trainable
                              )
    return backbone


def resnet_101(trainable: bool, catch_output: bool) -> ResnetBackbone:
    cells_number   = [3, 4, 23, 3]
    input_channels = [64, 256, 512, 1024]
    main_channels  = [64, 128, 256, 512]
    scale_rate     = 4
    cells          = resnet_cells(cells_number=cells_number,
                                  input_channels=input_channels,
                                  main_channels=main_channels,
                                  scale_rate=scale_rate,
                                  block_type="block_1",
                                  trainable=trainable)
    backbone = ResnetBackbone(obj_name="resnet_101",
                              cells=cells,
                              use_catch=catch_output,
                              trainable=trainable
                              )
    return backbone


def resnet_152(trainable: bool, catch_output: bool) -> ResnetBackbone:
    cells_number   = [3, 8, 36, 3]
    input_channels = [64, 256, 512, 1024]
    main_channels  = [64, 128, 256, 512]
    scale_rate     = 4
    cells          = resnet_cells(cells_number=cells_number,
                                  input_channels=input_channels,
                                  main_channels=main_channels,
                                  scale_rate=scale_rate,
                                  block_type="block_1",
                                  trainable=trainable)
    backbone = ResnetBackbone(obj_name="resnet_152",
                              cells=cells,
                              use_catch=catch_output,
                              trainable=trainable
                              )
    return backbone


def resnet_18(trainable: bool, catch_output: bool) -> ResnetBackbone:
    cells_number   = [2, 2, 2, 2]
    main_channels  = [64, 128, 256, 512]
    input_channels = [64, 64, 128, 256]
    scale_rate     = 1
    cells          = resnet_cells(cells_number=cells_number,
                                  input_channels=input_channels,
                                  main_channels=main_channels,
                                  scale_rate=scale_rate,
                                  block_type="block_2",
                                  trainable=trainable)
    backbone = ResnetBackbone(obj_name="resnet_18",
                              cells=cells,
                              use_catch=catch_output,
                              trainable=trainable
                              )
    return backbone


def resnet_34(trainable: bool, catch_output: bool) -> ResnetBackbone:
    cells_number   = [3, 4, 6, 3]
    main_channels  = [64, 128, 256, 512]
    input_channels = [64, 64, 128, 256]
    scale_rate     = 1
    cells          = resnet_cells(cells_number=cells_number,
                                  input_channels=input_channels,
                                  main_channels=main_channels,
                                  scale_rate=scale_rate,
                                  block_type="block_2",
                                  trainable=trainable)
    backbone = ResnetBackbone(obj_name="resnet_34",
                              cells=cells,
                              use_catch=catch_output,
                              trainable=trainable
                              )
    return backbone