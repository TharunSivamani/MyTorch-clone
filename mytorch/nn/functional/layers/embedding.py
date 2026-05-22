from mytorch import Tensor
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array

def auto_embedding(indices, weight):

    """
    [Vocab Size x Embedding Dim]
    
    """
    return weight[indices]


def embedding(indices, weight, fused=False):
    return auto_embedding(indices, weight)