"""N-dimensional arrays with explicit CPU/GPU placement.

This module defines :class:`Array`, a lightweight wrapper around
:class:`numpy.ndarray` on CPU and :class:`cupy.ndarray` on CUDA devices.
It mirrors common NumPy construction and elementwise semantics while
requiring operands in binary operations and NumPy ufuncs to reside on
the same device.

**Dependencies**

* **NumPy** — required; used for CPU storage and as the reference API.
* **CuPy** — optional; if importable, GPU arrays and ``device='cuda:N'``
  are supported. If CuPy is missing, only CPU mode is available.

**Module-level variables**

``CUDA_AVAILABLE`` : bool
    ``True`` if CuPy imported successfully and the runtime exposes CUDA.

``NUM_AVAILABLE_GPUS`` : int
    Number of CUDA devices reported by the runtime when CuPy is available;
    ``0`` otherwise.

**Device strings**

* ``\"cpu\"`` — data backed by NumPy.
* ``\"cuda\"`` or ``\"cuda:0\"`` — data on GPU index ``0`` (``cuda`` is
  normalized to ``cuda:0`` where applicable).
* ``\"cuda:N\"`` — data on GPU index ``N``.

**NumPy interoperability**

:class:`Array` implements ``__array_ufunc__`` and ``__array_function__`` so
calls like ``numpy.add(a, b)`` dispatch when arguments are compatible.
Mixed CPU/GPU inputs raise :exc:`RuntimeError` rather than performing
implicit host/device transfers.
"""
from __future__ import annotations

import warnings
import numpy as np

try:
    import cupy as cp
    CUDA_AVAILABLE = True
    NUM_AVAILABLE_GPUS = cp.cuda.runtime.getDeviceCount()
except ImportError:
    cp = None
    CUDA_AVAILABLE = False
    NUM_AVAILABLE_GPUS = 0
    warnings.warn("Cupy is not installed!")

class Array:
    """An N-D array with a fixed device (CPU or CUDA) and dtype.

    :class:`Array` wraps a NumPy or CuPy ndarray, tracks ``device`` and
    ``dtype``, and routes arithmetic and many NumPy ufuncs to the correct
    backend. Binary operators and in-place updates require the other
    operand's *declared* device to match when it is also an :class:`Array`
    or a CuPy/NumPy array with an inferable device.

    Attributes of the underlying buffer (e.g. ``shape``) are exposed as
    properties. Names not defined on :class:`Array` may be forwarded to the
    wrapped array via ``__getattr__``.

    Parameters
    ----------
    data : array_like, numpy.ndarray, cupy.ndarray, or Array
        Source values or buffer. Sequence and scalar inputs are converted
        with :func:`numpy.array` before placement. Existing ndarray buffers
        are moved or copied to ``device`` when it differs from their
        current location.
    device : str or None, optional
        Target device: ``\"cpu\"``, ``\"cuda\"``, ``\"cuda:N\"``, or
        ``None`` to infer from ``data`` (e.g. CuPy array → matching GPU, or
        CPU if no device information).
    dtype : str or numpy.dtype or None, optional
        Desired dtype as a string name (recommended) or ``None`` to choose
        from ``data``: ``float64``/``int64`` are narrowed to ``float32``/
        ``int32``; otherwise the dtype string is preserved. Mismatching
        buffers are cast after the array is placed on ``device``.

    Raises
    ------
    RuntimeError
        If a CUDA device is requested but CuPy is unavailable, the
        requested GPU index is out of range, or a move to GPU is
        requested without CUDA support.

    See Also
    --------
    Array.to : Move between devices.
    Array.zeros, Array.ones, Array.empty : Tensor factory methods.

    Notes
    -----
    Rich comparison, arithmetic, and in-place operators are bound at import
    time from the ``_binary_ufuncs``, ``_unary_ufuncs``, and ``_inplace_ops``
    tables at the bottom of this module.
    """

    _binary_ufuncs = {
        "__add__": "add", "__radd__": "add",
        "__sub__": "subtract", "__rsub__": "subtract",
        "__mul__": "multiply", "__rmul__": "multiply",
        "__truediv__": "true_divide", "__rtruediv__": "true_divide",
        "__floordiv__": "floor_divide", "__rfloordiv__": "floor_divide",
        "__matmul__": "matmul", "__rmatmul__": "matmul",
        "__pow__": "power", "__rpow__": "power",
        "__mod__": "remainder", "__rmod__": "remainder",
        "__and__": "bitwise_and", "__rand__": "bitwise_and",
        "__or__": "bitwise_or", "__ror__": "bitwise_or",
        "__xor__": "bitwise_xor", "__rxor__": "bitwise_xor",
        "__lt__": "less", "__le__": "less_equal",
        "__gt__": "greater", "__ge__": "greater_equal",
        "__eq__": "equal", "__ne__": "not_equal",
    }

    _inplace_ops = {
        "__iadd__": "add",
        "__isub__": "subtract",
        "__imul__": "multiply",
        "__itruediv__": "true_divide",
        "__ifloordiv__": "floor_divide",
        "__imatmul__": "matmul",
        "__ipow__": "power",
        "__imod__": "remainder",
        "__iand__": "bitwise_and",
        "__ior__": "bitwise_or",
        "__ixor__": "bitwise_xor",
    }

    _unary_ufuncs = {
        "__neg__": "negative",
        "__pos__": "positive",
        "__abs__": "absolute",
        "__invert__": "invert",
    }

    def __init__(self, data, device=None, dtype=None) -> None:
        """Construct an :class:`Array`; see the class docstring for semantics."""
        if device is not None:

            if device == "cpu":
                tgt_device = "cpu"
                tgt_device_idx = None

            elif "cuda" in device:
                if not CUDA_AVAILABLE:
                    raise RuntimeError("CUDA not supported, Install Cupy")
                
                tgt_device, tgt_device_idx = self.__parse_cuda_str(device_str = device)

                if tgt_device_idx + 1 > NUM_AVAILABLE_GPUS:
                    raise RuntimeError(f"cuda:{tgt_device_idx} does not exists")
                
        else:

            if hasattr(data, "device"):

                if isinstance(data.device, str):
                    if "cuda" in data.device:
                        tgt_device, tgt_device_idx = self.__parse_cuda_str(device_str = str(data.device))
                    else:
                        tgt_device, tgt_device_idx = "cpu", None

                elif isinstance(data.device, cp.cuda.device.Device):
                    tgt_device = "cuda"
                    tgt_device_idx = data.device.id

            else:
                tgt_device = "cpu"
                tgt_device_idx = None

        if dtype is None:

            if hasattr(data, "dtype"):
                current_dtype = str(data.dtype)
                if current_dtype == "float64":
                    dtype = "float32"
                elif current_dtype == "int64":
                    dtype = "int32"
                else:
                    dtype = current_dtype
            else:
                dtype = "float32"
        
        else:

            if not isinstance(dtype, str):
                dtype = str(dtype)


        if isinstance(data, (np.ndarray, cp.ndarray)):
            self._array = data
        else:
            self._array = np.array(data)

        src_dev = "cpu" if isinstance(self._array, np.ndarray) else f"cuda:{self._array.device.id}"

        self._array = self.__move_array(self._array, src_dev, tgt_device, tgt_device_idx)

        current_dtype = str(self._array.dtype)

        if current_dtype != dtype:
            if "cuda" in tgt_device and CUDA_AVAILABLE:
                with cp.cuda.Device(tgt_device_idx):
                    self._array = self._array.astype(dtype)
            
            else:
                self._array = self._array.astype(dtype)
        
        self._xp = np if isinstance(self._array, np.ndarray) else cp
        self._dev_id = None if self._xp is np else self._array.device.id
        self._device = "cpu" if self._xp is np else f"cuda:{self._dev_id}"

    @property
    def xp(self):
        """Module implementing elementwise ops for this array: ``numpy`` or ``cupy``."""
        return self._xp
    
    @property
    def device(self):
        """Placement string, e.g. ``\"cpu\"`` or ``\"cuda:0\"``."""
        return self._device
    
    @property
    def dtype(self):
        """NumPy dtype object of the underlying buffer."""
        return self._array.dtype

    @property
    def shape(self):
        """Tuple of axis lengths (same as the wrapped ndarray)."""
        return self._array.shape
    
    @property
    def ndim(self):
        """Number of dimensions."""
        return self._array.ndim
    
    @property
    def size(self):
        """Total number of elements (product of ``shape``)."""
        return self._array.size
    
    @property
    def T(self):
        """Transpose (view), same device as ``self``."""
        return Array(self._array.T, device=self._device)
    
    def astype(self, dtype):
        """Cast in-place and return ``self``.

        Parameters
        ----------
        dtype : str or numpy.dtype
            Target dtype name or descriptor.

        Returns
        -------
        Array
            ``self`` after updating the backing buffer.
        """
        if self.dtype == dtype:
            return self

        if self._xp is np:
            self._array = self._array.astype(dtype)
        
        else:
            with cp.cuda.Device(self._dev_id):
                self._array = self._array.astype(dtype)

        return self
    
    def to(self, device):
        """Return a copy of this array on ``device`` (or ``self`` if unchanged).

        Parameters
        ----------
        device : str
            ``\"cpu\"``, ``\"cuda\"``, or ``\"cuda:N\"``. Bare ``\"cuda\"`` is
            treated as ``\"cuda:0\"``.

        Returns
        -------
        Array
            New :class:`Array` sharing no storage with ``self`` when the
            device changes; otherwise ``self``.

        Raises
        ------
        RuntimeError
            If moving to CUDA is requested without a working CuPy/CUDA stack.
        """
        if device == "cuda":
            device = "cuda:0"
        
        if device == self._device:
            return self

        else:

            if device == "cpu":
                tgt_dev = "cpu"
                tgt_dev_idx = None
            else:
                tgt_dev, tgt_dev_idx = self.__parse_cuda_str(device)

            return Array(data=self.__move_array(arr=self._array, src_dev=self._device, tgt_dev=tgt_dev, tgt_dev_idx=tgt_dev_idx), device=device, dtype=self.dtype)

    def __parse_cuda_str(self, device_str):
        """Parse ``cuda`` / ``cuda:N`` into canonical ``(\"cuda\", index)``."""
        tgt_device = "cuda"
        tgt_device_idx = int(device_str.split(":")[-1]) if ":" in device_str else 0

        return tgt_device, tgt_device_idx
    
    def __move_array(self, arr, src_dev, tgt_dev, tgt_dev_idx=None):
        """Copy or view ``arr`` onto ``tgt_dev`` (internal)."""
        src_tgt = src_dev if src_dev == "cpu" else "cuda"
        src_idx = None if src_dev == "cpu" else int(src_dev.split(":")[-1])
        tgt_idx = tgt_dev_idx if tgt_dev == "cuda" else None

        if src_tgt == tgt_dev and src_idx == tgt_idx:
            return arr
        
        if tgt_dev == "cuda":
            if not CUDA_AVAILABLE:
                raise RuntimeError("Cuda is not supported!")
            
            if tgt_dev_idx is None:
                tgt_dev_idx = 0

            with cp.cuda.Device(tgt_dev_idx):
                return cp.asarray(arr)
            
        else:
            return cp.asarray(arr)

    def asnumpy(self):
        """Host-side NumPy view or copy.

        Returns
        -------
        numpy.ndarray
            The backing CPU array when ``device == \"cpu\"``; otherwise a
            newly allocated NumPy array with host data copied from GPU.
        """
        if self.device == "cpu":
            return self._array
        return cp.asnumpy(self._array)
    
    def _coerce_other(self, other):
        """Return ``(buffer, device_str_or_None)`` for binary/in-place ops."""
        if isinstance(other, Array):
            return other._array, other._device
        
        if isinstance(other, np.ndarray):
            return other, "cpu"

        if CUDA_AVAILABLE and isinstance(other, cp.ndarray):
            return other, f"cuda:{other.device.id}"
        
        return other, None
     
    @classmethod
    def _make_binary_op(cls, ufunc_name, reflect=False):
        """Build a binary dunder that dispatches to ``xp.<ufunc_name>``."""
        def op(self, other):
            other_arr, other_dev = self._coerce_other(other)
            
            if other_dev is not None and other_dev != self._device:
                raise RuntimeError(f"Expected all tensors to be on the "
                 f"same device, but found at least two devices, "
                 f"{self._device} and {other_dev}!")
            
            rhs = other_arr
            xp = self._xp
            func = getattr(xp, ufunc_name)

            if reflect:
                _in = (rhs, self._array)
            else:
                _in = (self._array, rhs)

            if xp is cp:
                with cp.cuda.Device(self._dev_id):
                    res = func(*_in)
            else:
                res = func(*_in)

            return Array(res, device=self._device)
        
        return op

    @classmethod
    def _make_unary_op(cls, ufunc_name):
        """Build a unary dunder that dispatches to ``xp.<ufunc_name>``."""
        def op(self):
            func = getattr(self._xp, ufunc_name)

            if self._xp is np:
                res = func(self._array)
            else:
                with cp.cuda.Device(self._dev_id):
                    res = func(self._array)
            return Array(res, device=self._device)
        return op

    @classmethod
    def _make_inplace_op(cls, ufunc_name):
        """Build an in-place dunder using ``ufunc(..., out=self._array)``."""
        def op(self, other):
            other_arr, other_dev = self._coerce_other(other)
            
            # print(other_dev, self._device)
            if other_dev is not None and other_dev != self._device:
                raise RuntimeError(f"Expected all tensors to be on the "
                 f"same device, but found at least two devices, "
                 f"{self._device} and {other_dev}!")
            
            func = getattr(self._xp, ufunc_name)

            if self._xp is np:
                func(self._array, other_arr, out=self._array)
            else:
                with cp.cuda.Device(self._dev_id):
                    func(self._array, other_arr, out=self._array)
            return self
        return op
    
    def __len__(self):
        """Length of the leading dimension (same as ``len(self._array)``)."""
        return len(self._array)
    
    def __repr__(self):
        """Human-readable preview: data, ``dtype``, and ``device`` on GPU."""
        data = self._array

        data_str = self._xp.array2string(
            data,
            separator=" ",
            precision=5,
            floatmode="fixed",
            max_line_width=80
        )

        # Indent continuation lines (like torch does)
        lines = data_str.split("\n")
        if len(lines) > 1:
            indent = " " * len("Array(")
            data_str = lines[0] + "\n" + "\n".join(indent + line for line in lines[1:])

        # Device info (only show if GPU)
        device_info = f", device='{self.device}'" if "cuda" in self.device else ""

        # Final string with dtype always showsn
        return f"Array({data_str}, dtype={self.dtype}{device_info})"

    def __array_function__(self, func, types, args, kwargs):
        """NumPy ``__array_function__`` dispatch when all types are :class:`Array`.

        Unwraps arguments to the underlying ndarray types, checks a single
        device, invokes the matching ``numpy`` or ``cupy`` callable named
        like ``func``, and re-wraps ndarray results.

        Parameters
        ----------
        func : Callable
            The NumPy API function being invoked.
        types : tuple[type, ...]
            Argument types participating in dispatch.
        args : tuple
            Positional arguments to ``func``.
        kwargs : dict
            Keyword arguments to ``func``.

        Returns
        -------
        Array or Any
            :class:`Array` when the implementation returns an ndarray;
            otherwise the raw return value. Returns ``NotImplemented`` to
            defer to other types when this implementation does not apply.

        Raises
        ------
        RuntimeError
            If arguments would require mixing devices.
        """
        if not all(issubclass(t, Array) for t in types):
            return NotImplemented
        
        devices = set()

        def handler(x):
            if isinstance(x, Array):
                devices.add(x._device)
                return x._array
            elif isinstance(x, (list, tuple)):
                return type(x)(handler(y) for y in x)
            elif isinstance(x, dict):
                return {k: handler(v) for k, v in x.items()}
            else:
                return x
            
        handled_args = handler(args)
        handled_kwargs = handler(kwargs)

        if len(devices) > 1:
            raise RuntimeError("Expected all tensors to be on the same device, but found at least two devices!")
        
        if not devices:
            device = self._device
        else:
            device = list(devices)[0]

        xp = cp if "cuda" in device else np

        xp_func = getattr(xp, func.__name__, None)

        if xp_func is None:
            return NotImplemented
        
        if "cuda" in device:
            _, dev_id = self.__parse_cuda_str(device)
            with cp.cuda.Device(dev_id):
                result = xp_func(*handled_args, **handled_kwargs)
        else:
            result = xp_func(*handled_args, **handled_kwargs)

        if isinstance(result, (np.ndarray, cp.ndarray if CUDA_AVAILABLE else type(None))):
            return Array(result, device=device)
        
        return result

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """NumPy ufunc override (NEP 13): run ufuncs on one device and wrap results.

        Parameters
        ----------
        ufunc : numpy.ufunc
            The ufunc instance (e.g. ``numpy.add``).
        method : str
            Ufunc method name (e.g. ``\"__call__\"``, ``\"reduce\"``).
        *inputs
            Operands; :class:`Array` instances are unwrapped to their buffers.
        **kwargs
            Forwarded to the ufunc (e.g. ``out=``).

        Returns
        -------
        Array or Any
            :class:`Array` for ndarray results; otherwise the ufunc return
            value (e.g. scalars).

        Raises
        ------
        RuntimeError
            If operands span more than one device.
        """
        arrays = []
        devices = set()

        # Extract underlying arrays and track devices
        for x in inputs:
            if isinstance(x, Array):
                arrays.append(x._array)
                devices.add(x._device)
            else:
                arrays.append(x)
                if isinstance(x, np.ndarray):
                    devices.add("cpu")
                elif CUDA_AVAILABLE and isinstance(x, cp.ndarray):
                    devices.add(f"cuda:{x.device.id}")
     
        # Enforce single-device rule
        if len(devices) > 1:
            raise RuntimeError(f"All inputs must be on the same device, found: {devices}")

        # Pick device
        device = list(devices)[0] if devices else "cpu"

        # Run the ufunc under the correct context
        if "cuda" in device:
            _, dev_id = self.__parse_cuda_str(device)
            with cp.cuda.Device(dev_id):
                result = getattr(ufunc, method)(*arrays, **kwargs)
        else:
            result = getattr(ufunc, method)(*arrays, **kwargs)

        # Wrap result back into Array if applicable
        if isinstance(result, (np.ndarray, cp.ndarray)):
            return Array(result, device=device)
        return result

    def __getitem__(self, idx):
        """Index or slice the buffer.

        :class:`Array` indices (including nested in tuples) are unwrapped to
        their underlying ndarray for advanced indexing.

        Returns
        -------
        Array or numpy.ndarray or scalar
            On GPU, the result is wrapped in a new :class:`Array` on the same
            device. On CPU, NumPy's indexing rules apply and the return value
            may be a view, ndarray, or scalar without wrapping.
        """
        def _coerce_index(index):
            if isinstance(index, tuple):
                return tuple(_coerce_index(i) for i in index)
            
            if isinstance(index, Array):
                return index._array
            
            if hasattr(index, "data") and isinstance(index.data, Array):
                return index.data._array
            
            return index
        
        idx = _coerce_index(idx)

        if self._xp is np:
            return self._array[idx]
        else:
            with cp.cuda.Device(self._array.device.id):
                result = self._array[idx]

        return Array(result, device=self._device)

    def __setitem__(self, idx, value):
        """Write ``value`` into ``self._array[idx]`` (unwraps ``Array`` values)."""
        if isinstance(value, Array):
            value = value._array
        self._array[idx] = value

    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped ndarray (e.g. ``reshape``)."""
        if hasattr(self._array, name):
            attr = getattr(self._array, name)
            return attr
        raise AttributeError(f"'Array' object has no attribute '{name}'")

    @classmethod
    def _wrap_factory(cls, xp_func, *args, device="cpu", dtype="float32", **kwargs):
        """Call ``numpy.<xp_func>`` or ``cupy.<xp_func>`` and wrap the result."""
        xp = np if "cpu" in device else cp

        _, tgt_device = ("cpu", None)

        if "cuda" in device:
            _, tgt_device_idx = cls.__parse_cuda_str(device)

        if xp == cp:
            with cp.cuda.Device(tgt_device_idx):
                arr = getattr(xp, xp_func)(*args, **kwargs)
        else:
            arr = getattr(xp, xp_func)(*args, **kwargs)

        if dtype is not None:
            current_dtype = str(arr.dtype)
            if current_dtype != dtype:
                if xp == np:
                    arr = arr.astype(dtype)
                else:
                    with cp.cuda.Device(tgt_device_idx):
                        arr = arr.astype(dtype)

        return cls(arr, device=device, dtype=str(arr.dtype))

    @classmethod
    def zeros(cls, shape, device="cpu", dtype="float32"):
        """Return a new array of zeros with given ``shape`` on ``device``."""
        return cls._wrap_factory("zeros", shape, device=device, dtype=dtype)
    
    @classmethod
    def ones(cls, shape, device="cpu", dtype="float32"):
        """Return a new array of ones with given ``shape`` on ``device``."""
        return cls._wrap_factory("ones", shape, device=device, dtype=dtype)

    @classmethod
    def empty(cls, shape, device="cpu", dtype="float32"):
        """Return an uninitialized array (like :func:`numpy.empty`)."""
        return cls._wrap_factory("empty", shape, device=device, dtype=dtype)

    @classmethod
    def full(cls, shape, fill_value, device="cpu", dtype="float32"):
        """Return an array filled with ``fill_value`` (like :func:`numpy.full`)."""
        return cls._wrap_factory("full", shape, fill_value, device=device, dtype=dtype)

    @classmethod
    def arange(cls, start, end=None, step=1, device="cpu", dtype="float32"):
        """1-D evenly spaced values (like :func:`numpy.arange`).

        If ``end`` is omitted, ``start`` is treated as the stop and zero as start.
        """
        if end is None:
            end = start
            start = 0

        return cls._wrap_factory("arange", start, end, step, device=device, dtype=dtype)

    @classmethod
    def linspace(cls, start, end, num=50, device="cpu", dtype="float32"):
        """``num`` evenly spaced samples over ``[start, end]`` (inclusive)."""
        xp = np if "cpu" in device else cp
        arr = xp.linspace(start, end, num=num, dtype=dtype)
        return cls(arr, device=device, dtype=str(arr.dtype))

    @classmethod
    def eye(cls, N, M=None, k=0, device="cpu", dtype="float32"):
        """Identity-like 2-D matrix with ones on the ``k``-th diagonal."""
        return cls._wrap_factory("eye", N, M, k, device=device, dtype=dtype)

    @classmethod
    def randn(cls, shape, device="cpu", dtype="float32"):
        """Standard normal samples with shape ``shape`` (like ``xp.random.randn``)."""
        xp = np if "cpu" in device else cp
        tgt_device_idx = None
        
        if "cuda" in device:
           _, tgt_device_idx = cls(None).__parse_cuda_str(device)
        
        # Generate array on the correct device
        if xp is cp:
            with cp.cuda.Device(tgt_device_idx):
                arr = xp.random.randn(*shape).astype(dtype)
        else:
            arr = xp.random.randn(*shape).astype(dtype)
        
        return cls(arr, device=device, dtype=str(arr.dtype))
    
    @classmethod
    def rand(cls, shape, device="cpu", dtype="float32"):
        """Uniform ``[0, 1)`` samples with shape ``shape`` (like ``xp.random.rand``)."""
        xp = np if "cpu" in device else cp
        tgt_device_idx = None

        if "cuda" in device:
            _, tgt_device_idx = cls(None).__parse_cuda_str(device)

        # Generate array on the correct device
        if xp is cp:
            with cp.cuda.Device(tgt_device_idx):
                arr = xp.random.rand(*shape).astype(dtype)
        else:
            arr = xp.random.rand(*shape).astype(dtype)

        return cls(arr, device=device, dtype=str(arr.dtype))

    @classmethod
    def randint(cls, low, high, shape, device="cpu", dtype="int32"):
        """Random integers in ``[low, high)`` with given ``shape``."""
        xp = np if "cpu" in device else cp
        arr = xp.random.randint(low, high, size=shape, dtype=dtype)
        return cls(arr, device=device, dtype=str(arr.dtype))

    @classmethod
    def tril(cls, x, k=0, device="cpu", dtype="float32"):
        """Lower triangle of ``x`` (see :func:`numpy.tril`)."""
        return cls._wrap_factory("tril", x, k=k, device=device, dtype=dtype)
 
    @classmethod
    def zeros_like(cls, other, device=None, dtype=None):
        """Zeros with the same shape (and default dtype/device) as ``other``."""
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls._wrap_factory("zeros_like", other, device=device, dtype=dtype)

    @classmethod
    def ones_like(cls, other, device=None, dtype=None):
        """Ones with the same shape (and default dtype/device) as ``other``."""
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls._wrap_factory("ones_like", other, device=device, dtype=dtype)

    @classmethod
    def empty_like(cls, other, device=None, dtype=None):
        """Uninitialized array matching ``other``'s shape and defaults."""
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls._wrap_factory("empty_like", other, device=device, dtype=dtype)
    
    @classmethod
    def full_like(cls, other, fill_value, device=None, dtype=None):
        """Array filled with ``fill_value``, matching ``other``'s shape."""
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls._wrap_factory("full_like", other, fill_value, device=device, dtype=dtype)
    
    @classmethod
    def randn_like(cls, other, device=None, dtype=None):
        """Standard normal tensor with the same ``shape`` as ``other``."""
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls.randn(other.shape, device=device, dtype=dtype)
    
    @classmethod
    def rand_like(cls, other, device=None, dtype=None):
        """Uniform ``[0,1)`` tensor with the same ``shape`` as ``other``."""
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls.rand(other.shape, device=device, dtype=dtype)


# --- Dynamic operator binding ---
# Binary, unary, and in-place dunders are attached here so ``Array`` stays
# declarative; see ``_binary_ufuncs``, ``_unary_ufuncs``, and ``_inplace_ops``.
for dunder, ufunc in Array._binary_ufuncs.items():
    reflect = dunder.startswith("__r")
    setattr(Array, dunder, Array._make_binary_op(ufunc, reflect=reflect))

for dunder, ufunc in Array._unary_ufuncs.items():
    setattr(Array, dunder, Array._make_unary_op(ufunc))

for dunder, ufunc in Array._inplace_ops.items():
    setattr(Array, dunder, Array._make_inplace_op(ufunc))


if __name__ == "__main__":
    def test_array_operations():
        # --- CPU arrays ---
        a_cpu = Array([1,2,3], device="cpu")
        b_cpu = Array([4,5,6], device="cpu")

        # Binary operations (add, sub, mul)
        assert np.allclose((a_cpu + b_cpu).asnumpy(), np.array([5,7,9]))
        assert np.allclose((b_cpu - a_cpu).asnumpy(), np.array([3,3,3]))
        assert np.allclose((a_cpu * b_cpu).asnumpy(), np.array([4,10,18]))

        # Unary operations
        assert np.allclose((-a_cpu).asnumpy(), np.array([-1,-2,-3]))
        assert np.allclose((+b_cpu).asnumpy(), np.array([4,5,6]))
        assert np.allclose(abs(Array([-1,-2,-3])).asnumpy(), np.array([1,2,3]))

        # Inplace operations
        a_copy = Array([1,2,3])
        a_copy += Array([10,20,30])
        assert np.allclose(a_copy.asnumpy(), np.array([11,22,33]))

        # --- GPU arrays ---
        if CUDA_AVAILABLE:
            a_gpu = Array([1,2,3], device="cuda:0")
            b_gpu = Array([4,5,6], device="cuda:0")

            # Binary ops
            assert np.allclose((a_gpu + b_gpu).asnumpy(), np.array([5,7,9]))
            assert np.allclose((b_gpu - a_gpu).asnumpy(), np.array([3,3,3]))
            assert np.allclose((a_gpu * b_gpu).asnumpy(), np.array([4,10,18]))

            # Unary ops
            assert np.allclose((-a_gpu).asnumpy(), np.array([-1,-2,-3]))
            assert np.allclose((+b_gpu).asnumpy(), np.array([4,5,6]))

            # Inplace ops
            a_copy_gpu = Array([1,2,3], device="cuda:0")
            a_copy_gpu *= Array([10,20,30], device="cuda:0")
            assert np.allclose(a_copy_gpu.asnumpy(), np.array([10,40,90]))

            # Device mismatch errors
            try:
                _ = a_cpu + a_gpu
            except RuntimeError as e:
                assert "device" in str(e)

            try:
                _ = np.add(a_cpu, a_gpu)
            except RuntimeError as e:
                assert "device" in str(e)

            # Using .xp backend
            assert np.allclose(np.add(a_gpu, a_gpu).asnumpy(), np.array([2,4,6]))

        # Factory functions
        z = Array.zeros((2,2), device="cpu")
        assert np.allclose(z.asnumpy(), np.zeros((2,2)))

        o = Array.ones((2,2), device="cpu")
        assert np.allclose(o.asnumpy(), np.ones((2,2)))

        r = Array.arange(3, device="cpu")
        assert np.allclose(r.asnumpy(), np.array([0,1,2]))

        print("All CPU/GPU tests passed!")

   
    test_array_operations()