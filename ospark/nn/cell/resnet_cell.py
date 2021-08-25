from ospark.nn.cell import Cell
from ospark.nn.block.resnet_block import ResnetBlock


class ResnetCell(Cell):

    def __init__(self,
                 obj_name: str,
                 block_cls: ResnetBlock,
                 block_number: int,
                 scale_rate: int,
                 input_channel: int,
                 trainable: bool):
        super().__init__(obj_name=obj_name)
        self._block_cls     = block_cls
        self._block_number  = block_number
        self._scale_rate    = scale_rate
        self._input_channel = input_channel
        self._trainable     = trainable