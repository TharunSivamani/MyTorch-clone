import numpy as np
from mytorch import Tensor
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array

def auto_relu(input):
    
    mask = Tensor(np.where(input.data < 0, 0, 1).astype(input.dtype, copy=False))
    return mask * input

def manual_relu(input):
    
    input_arr = get_inner_array(input)
    mask = (input_arr < 0)
    input_arr[mask] = 0

    def _relu_backward(output_grad):
        if input.requires_grad:

            output_grad[mask] = 0

            if input.grad is None:
                input.grad = output_grad
            else:
                input.grad += output_grad

    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        input_arr,
        requires_grad=requires_grad,
        grad_fn=_relu_backward if requires_grad else None,
        grad_fn_name="<ReLUBackward>" if requires_grad else None,
        device=input.device,
        dtype=input.dtype
    )

    if requires_grad:
        out._add_parents(input)

    return out

def relu(input, auto=False, fused=False):

    if auto:
        return auto_relu(input)
    else:
        return manual_relu(input)