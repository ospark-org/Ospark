from __future__ import annotations
from ospark.nn.cell import Cell
from ospark.nn.block.resnet_block import ResnetBlock, BottleneckBuildingBlock, BuildingBlock
from typing import List, NoReturn
import tensorflow as tf


class ResnetCell(Cell):

    def __init__(self,
                 obj_name: str,
                 blocks: List[ResnetBlock],
                 is_training: bool):
        super().__init__(obj_name=obj_name, is_training=is_training)
        self._blocks    = blocks

    @property
    def blocks(self) -> List[ResnetBlock]:
        return self._blocks

    @classmethod
    def build_cell(cls,
                   obj_name: str,
                   block_type: str,
                   block_number: int,
                   scale_rate: int,
                   input_channel: int,
                   main_channel: int,
                   strides: int,
                   is_training: bool
                   ) -> ResnetCell:
        blocks = []
        if block_type == "block_1":
            block_cls = BottleneckBuildingBlock
        elif block_type == "block_2":
            block_cls = BuildingBlock
        else:
            raise KeyError("block_type is only block_1 and block_2, please check block_type")

        for i in range(block_number):
            name = f"block_{i}"
            if i == 0:
                block = block_cls(obj_name=name,
                                  input_channel=input_channel,
                                  main_channel=main_channel,
                                  scale_rate=scale_rate,
                                  strides=strides,
                                  use_shortcut_conv=True,
                                  is_training=is_training)
            else:
                block = block_cls(obj_name=name,
                                  input_channel=input_channel,
                                  main_channel=main_channel,
                                  scale_rate=scale_rate,
                                  strides=1,
                                  is_training=is_training)
            blocks.append(block)
        return cls(obj_name=obj_name, blocks=blocks, is_training=is_training)

    def in_creating(self) -> NoReturn:
        for block in self.blocks:
            self.assign(component=block)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        output = input_data
        for block in self.blocks:
            output = block.pipeline(output)
        return output


def resnet_cells(cells_number: List[int],
                 input_channels: List[int],
                 main_channels: List[int],
                 scale_rate: int,
                 block_type: str,
                 is_training: bool) -> List[ResnetCell]:
    cells = []
    for i, parameters in enumerate(zip(cells_number, input_channels, main_channels)):
        block_number, input_channel, main_channel = parameters
        name = f"cell_{i}"
        if i == 0:
            cell = ResnetCell.build_cell(obj_name=name,
                                         block_type=block_type,
                                         block_number=block_number,
                                         scale_rate=scale_rate,
                                         input_channel=input_channel,
                                         main_channel=main_channel,
                                         strides=1,
                                         is_training=is_training
                                         )
        else:
            cell = ResnetCell.build_cell(obj_name=name,
                                         block_type=block_type,
                                         block_number=block_number,
                                         scale_rate=scale_rate,
                                         input_channel=input_channel,
                                         main_channel=main_channel,
                                         strides=2,
                                         is_training=is_training
                                         )
        cells.append(cell)
    return cells

