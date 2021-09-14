from ospark.former.transformer import Transformer
from ospark.former.informer import Informer
from ospark.former.former import Former

def transformer(block_number: int,
                ) -> Former:
    return Transformer.quick_build(block_number=block_number,
                                   )

def exdeep_transformer() -> Former:
    return Transformer.quick_build_exdeep()

def informer() -> Former:
    return Informer.quick_build()