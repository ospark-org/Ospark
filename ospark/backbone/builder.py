from .resnet import ResnetBackbone
from ospark.nn.cell.resnet_cell import resnet_cells


def resnet50(is_training: bool, catch_output: bool) -> ResnetBackbone:
    cells_number   = [3, 4, 6, 3]
    input_channels = [64, 256, 512, 1024]
    main_channels  = [64, 128, 256, 512]
    scale_rate     = 4
    cells          = resnet_cells(cells_number=cells_number,
                                  input_channels=input_channels,
                                  main_channels=main_channels,
                                  scale_rate=scale_rate,
                                  block_type="block_1",
                                  is_training=is_training)
    backbone = ResnetBackbone(obj_name="resnet_50",
                              cells=cells,
                              is_keep_blocks=catch_output,
                              is_training=is_training
                              )
    return backbone


def resnet101(is_training: bool, catch_output: bool) -> ResnetBackbone:
    cells_number   = [3, 4, 23, 3]
    input_channels = [64, 256, 512, 1024]
    main_channels  = [64, 128, 256, 512]
    scale_rate     = 4
    cells          = resnet_cells(cells_number=cells_number,
                                  input_channels=input_channels,
                                  main_channels=main_channels,
                                  scale_rate=scale_rate,
                                  block_type="block_1",
                                  is_training=is_training)
    backbone = ResnetBackbone(obj_name="resnet_101",
                              cells=cells,
                              is_keep_blocks=catch_output,
                              is_training=is_training
                              )
    return backbone


def resnet152(is_training: bool, catch_output: bool) -> ResnetBackbone:
    cells_number   = [3, 8, 36, 3]
    input_channels = [64, 256, 512, 1024]
    main_channels  = [64, 128, 256, 512]
    scale_rate     = 4
    cells          = resnet_cells(cells_number=cells_number,
                                  input_channels=input_channels,
                                  main_channels=main_channels,
                                  scale_rate=scale_rate,
                                  block_type="block_1",
                                  is_training=is_training)
    backbone = ResnetBackbone(obj_name="resnet_152",
                              cells=cells,
                              is_keep_blocks=catch_output,
                              is_training=is_training
                              )
    return backbone


def resnet18(is_training: bool, catch_output: bool) -> ResnetBackbone:
    cells_number   = [2, 2, 2, 2]
    main_channels  = [64, 128, 256, 512]
    input_channels = [64, 64, 128, 256]
    scale_rate     = 1
    cells          = resnet_cells(cells_number=cells_number,
                                  input_channels=input_channels,
                                  main_channels=main_channels,
                                  scale_rate=scale_rate,
                                  block_type="block_2",
                                  is_training=is_training)
    backbone = ResnetBackbone(obj_name="resnet_18",
                              cells=cells,
                              is_keep_blocks=catch_output,
                              is_training=is_training
                              )
    return backbone


def resnet34(is_training: bool, catch_output: bool) -> ResnetBackbone:
    cells_number   = [3, 4, 6, 3]
    main_channels  = [64, 128, 256, 512]
    input_channels = [64, 64, 128, 256]
    scale_rate     = 1
    cells          = resnet_cells(cells_number=cells_number,
                                  input_channels=input_channels,
                                  main_channels=main_channels,
                                  scale_rate=scale_rate,
                                  block_type="block_2",
                                  is_training=is_training)
    backbone = ResnetBackbone(obj_name="resnet_34",
                              cells=cells,
                              is_keep_blocks=catch_output,
                              is_training=is_training
                              )
    return backbone