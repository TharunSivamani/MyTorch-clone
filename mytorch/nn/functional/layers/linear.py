"""
Forward:
    y = x @ W.T + b
    x: (B, I)        input
    W: (O, I)        weight
    b: (O,)          bias (broadcasted)
    y: (B, O)        output

Backward (given grad_y = ∂L/∂y ∈ (B, O)):

    dx = grad_y @ W
          (B, O) @ (O, I) -> (B, I)

    dW = (grad_y.T @ x).T
          (O, B) @ (B, I) -> (O, I)
        implemented as: np.matmul(grad_y.T, x).T
        or equivalently: np.matmul(x.T, grad_y).T

    db = grad_y.sum(axis=0)
          (B, O) -> (O,)


CAVEAT:
Our fused_grouped_matmul is only as fast as CUDNN IF Triton Autotune is enabled. Normally we dont 
care about this, but in some problems, like Transformers where every batch can be a different sequence length
the change of shape will retrigger the Triton autotuner which is very slow. So we add an extra environment flag
here to disable the fused_grouped_matmul and just use normal CUPY matmuls that are already have a decision 
pathway of kernel settings to use provided by vendors! This is a little hacky but its fine! We also ensure that
in the fused_linear, if an activation function was passed in to be fused, it wont be fused anymore but
will still occur in the forward pass separately!
"""

import os
import numpy as np
from mytorch import Tensor
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array

def reshape_for_linear(x):

    """
    Linear layers can pass in multidim tensors, for example
    if our data is:
    
    [A x B x C x I], we rehsape it to [A * B * C x I], perform the
    matmul, and return back to the original shape
    """

    reshaped = False
    *dims, in_features = x.shape

    if len(dims) > 1:
        reshaped = True

    if reshaped:
        x = x.reshape(np.prod(dims), in_features)
    
    return x, dims, reshaped

def auto_linear(input, weight, bias=None, *args):
    
    inputs, dims, reshaped_flag = reshape_for_linear(input)

    # w: (O, I)
    out_features = weight.shape[0]

    output = input @ weight.transpose(-1, -2) # (B, O)

    if bias is not None:
        output = output + bias.reshape(1, -1) # # (B, O) + (1, O)

    if reshaped_flag:
        output = output.reshape(*dims, out_features)

    return output  

def manual_linear(input, weight, bias=None, *args):
    
    input, dims, reshaped_flag = reshape_for_linear(input)
    out_features, in_features = weight.shape

    input_arr = get_inner_array(input)
    weight_arr = get_inner_array(weight).T

    if bias is not None:
        bias_arr = get_inner_array(bias)
    
    output = np.matmul(input_arr, weight_arr)
    if bias is not None:
        output = output + bias_arr.reshape(1, -1)
    
    if reshaped_flag:
        output = output.reshape(*dims, out_features)

    def _linear_backward(grad_output):

        # Our gradients are coming in the shape of (A x B x C x O)
        # But our operation happened in the shape of (N x O)
        # So change our grad_output shape to that by flattening

        if reshaped_flag:
            grad_output = grad_output.reshape(-1, out_features) # (A*B*C, O)
        
        if weight.requires_grad:
            # input_arr (N x I), grad_output (N, O) -> (I, O).T -> (O, I) (shape of my weights)
            grad_W = np.matmul(input_arr.T, grad_output) # Upstream part

            if weight.grad is None:
                weight.grad = grad_W.T
            else:
                weight.grad += grad_W.T
            grad_W = None

        if bias is not None and bias.requires_grad:
            grad_b = grad_output.sum(axis=0)
            if bias.grad is None:
                bias.grad = grad_b
            else:
                bias.grad += grad_b

        if input.requires_grad:

            grad_input = np.matmul(grad_output, weight_arr.T)
            grad_input = grad_input.reshape(*dims, in_features)

            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input
            grad_input = None

    requires_grad = input.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    requires_grad = requires_grad and Tensor.build_graph_enabled()

    output = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_linear_backward if requires_grad else None,
        grad_fn_name="<LinearBackward" if requires_grad else None
    )

    if requires_grad:
        output._add_parents(input, weight, bias)
    
    return output

def linear(input, weight, bias=None, auto=False, fused=False, act_func=None):

    if auto:
        return auto_linear(input, weight, bias)
    else:
        return manual_linear(input, weight, bias)


def fused_linear(input, weight, bias=None, act_func=None):
    pass