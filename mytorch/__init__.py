import os
import cupy as cp

device = cp.cuda.Device()

cc_major, cc_minor = device.compute_capability

if int(cc_major) >= 8:
    os.environ["CUPY_TF32"] = "1"

import warnings
warnings.filterwarnings("ignore", module="pydantic")

from mytorch.tensor import Tensor, no_grad, zeros, ones, empty, full, \
    arange, linspace, eye, tril, randn, rand, randint, zeros_like, \
        ones_like, empty_like, randn_like, rand_like, full_like

