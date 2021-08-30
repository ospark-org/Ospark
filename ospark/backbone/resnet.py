from ospark.nn.cell.resnet_cell import ResnetCell
from ospark.nn.component.basic_module import BasicModule
from ospark.nn.layer.convolution_layer import ConvolutionLayer
from typing import List, NoReturn, Optional
from ospark.nn.cell.resnet_cell import resnet_cells
import tensorflow as tf

class Resnet(BasicModule):

    def __init__(self,
                 obj_name: str,
                 cells: List[ResnetCell],
                 trainable: Optional[bool]=True
                 ):
        super().__init__(obj_name=obj_name)
        self._cells              = cells
        self._trainable          = trainable

    @property
    def cells(self) -> List[ResnetCell]:
        return self._cells

    @property
    def trainable(self) -> bool:
        return self._trainable

    def initialize(self) -> NoReturn:
        for cell in self.cells:
            self.assign(component=cell)
        self.assign(component=ConvolutionLayer.bn_relu_conv(obj_name="first_conv",
                                                            filter_size=[7, 7, 3, 64],
                                                            strides=[1, 2, 2, 1],
                                                            padding="SAME",
                                                            trainable=self.trainable))

    def model(self, input_data: tf.Tensor) -> tf.Tensor:
        output = self.assigned.first_conv(input_data)
        output = tf.nn.max_pool2d(output, strides=2, ksize=3, padding="SAME")
        for cell in self.cells:
            output = cell(output)
        return output

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        output = self.model(input_data=input_data)
        return output


def resnet_50(trainable: bool):
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
    model = Resnet(obj_name="resnet_50",
                   cells=cells,
                   trainable=trainable
                   )
    return model


def resnet_101(trainable: bool):
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
    model = Resnet(obj_name="resnet_50",
                   cells=cells,
                   trainable=trainable
                   )
    return model


def resnet_152(trainable: bool):
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
    model = Resnet(obj_name="resnet_50",
                   cells=cells,
                   trainable=trainable
                   )
    return model


def resnet_18(trainable: bool):
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
    model = Resnet(obj_name="resnet_50",
                   cells=cells,
                   trainable=trainable
                   )
    return model


def resnet_34(trainable: bool):
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
    model = Resnet(obj_name="resnet_50",
                   cells=cells,
                   trainable=trainable
                   )
    return model