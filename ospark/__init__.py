__version__ = "0.0.1"
__name__ = "ospark"

import ospark
from ospark.nn.component.weight import Weight, WeightOperator
from ospark.nn.component import weight
from ospark.nn.component import activation
from ospark.utility import weight_initializer
from ospark.nn.layers import Layer, normalization
from ospark.nn.block import Block
from ospark.nn.cell import Cell
from ospark.nn.model import Model