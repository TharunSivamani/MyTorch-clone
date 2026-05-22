import numpy as np
from mytorch import Tensor
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array

def auto_cross_entropy(logits, targets, ignore_index=-100, *args):
    
    """
    Automatic diff through CrossEntropy seems to have some instability, 
    so we are sticking to manual diff and fused diff
    """
    raise NotImplementedError

def manual_cross_entropy(logits, targets, ignore_index=-100, *args):

    *dims, num_classes = logits.shape
    flattened_dim = np.prod(dims)
    
    logits_data = get_inner_array(logits).reshape(flattened_dim, num_classes).astype("float32", copy=False)
    logits_data = logits_data.reshape(flattened_dim, num_classes).astype("float32", copy=False)
    targets_data = get_inner_array(targets).reshape(flattened_dim)

    mask = (targets_data != ignore_index)
    valid_counts = mask.sum()

    # Stable logsumexp per row
    logits_max = np.max(logits_data, axis=1, keepdims=True)
    exp_shifted = np.exp(logits_data - logits_max)  # shape (B, C)
    logsumexp = np.log(np.sum(exp_shifted, axis=1, keepdims=True)) + logits_max  # shape (B, 1)

    # Negative log-likelihood only for valid rows
    nll = (logsumexp.flatten() - logits_data[np.arange(flattened_dim), targets_data]) * mask
    loss_value = np.sum(nll) / valid_counts

    loss_value = loss_value.astype(logits.dtype, copy=False)
    
    def _cross_entropy_backward(grad_output):

        if logits.requires_grad:
        
            # Compute softmax probabilities for all rows
            softmax = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)  # shape (B, C)
            
            # Initialize gradient
            grad_input = softmax.copy()
            grad_input[np.arange(flattened_dim), targets_data] -= 1  # softmax - one_hot

            # Scale by grad_output and divide by valid counts
            grad_input *= (grad_output / valid_counts)
            
            # Zero out ignored rows
            grad_input *= mask.reshape(-1,1)

            # Reshape back to original logits shape and dtype
            grad_input = grad_input.reshape(logits.shape).astype(logits.dtype, copy=False)

            if logits.grad is None:
                logits.grad = grad_input
            else:
                logits.grad += grad_input

    requires_grad = logits.requires_grad
    requires_grad = requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        loss_value,
        requires_grad=requires_grad,
        grad_fn=_cross_entropy_backward if requires_grad else None,
        grad_fn_name="<CrossEntropyBackward>" if requires_grad else None
    )
    
    if requires_grad:
        out._add_parents(logits)

    return out

def cross_entropy(logits, targets, ignore_index=-100, softcap=None, fused=False):
    return manual_cross_entropy(logits, targets, ignore_index)
