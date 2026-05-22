import numpy as np
from mytorch import Tensor
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array

def auto_softmax(x, dim=-1):
    
    """
    e^x / sum(e^j) -> e^(x-max) / sum(e^(j-max))
    """

    max_x = x.max(dim=dim, keepdims=True)
    x_shifted = x - max_x
    exp_x = x_shifted.exp()
    sum_exp = exp_x.sum(dim=dim, keepdims=True)
    return exp_x / sum_exp

def manual_softmax(x, dim=-1):

    x_arr = get_inner_array(x)

    max_val = np.max(x_arr, axis=dim, keepdims=True)
    shifted = x_arr - max_val
    exp_x = np.exp(shifted)
    sum_exp = np.sum(exp_x, axis=dim, keepdims=True)
    out_data = exp_x / sum_exp

    def _softmax_backward(grad_output):

        if x.requires_grad:
            # s * (grad - sum(grad * s)) - non jacobian form
            sum_grad_s = np.sum(grad_output * out_data, axis=dim, keepdims=True)
            grad_input = out_data * (grad_output - sum_grad_s)

            if x.grad is None:
                x.grad = grad_input
            else:
                x.grad += grad_input
    
    requires_grad = x.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        out_data,
        requires_grad=requires_grad,
        grad_fn=_softmax_backward if requires_grad else None,
        grad_fn_name="<SoftmaxBackward>" if requires_grad else None,
    )

    if requires_grad:
        out._add_parents(x)
    
    return out

def softmax(x, dim=-1, auto=False, fused=False):
    if auto:
        return auto_softmax(x, dim=dim)
    else:
        return manual_softmax(x, dim=dim)