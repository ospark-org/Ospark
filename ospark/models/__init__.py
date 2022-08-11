from ospark.models.transformer import Transformer
from ospark.models.informer import Informer
from ospark.models.former import Former

def transformer(block_number: int) -> Former:
    return Transformer.quick_build(block_number=block_number)

def exdeep_transformer() -> Former:
    return Transformer.quick_build_exdeep()

def informer() -> Former:
    return Informer.quick_build()