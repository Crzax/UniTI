import numpy as np


__device_name__ = "numpy"
_datatype = np.float32
_datetype_size = np.dtype(_datatype).itemsize


class Array:
    def __init__(self, size):
        self.array = np.empty(size, dtype=np.float32)

    @staticmethod
    def from_flat(flat_array):
        """Wrap an existing flat float32 numpy array without copying."""
        arr = Array.__new__(Array)
        arr.array = flat_array
        return arr

    @property
    def size(self):
        return self.array.size


def to_numpy(a, shape, strides, offset):
    return np.lib.stride_tricks.as_strided(
        a.array[offset:], shape, tuple([s * _datetype_size for s in strides])
    )


def from_numpy(a, out):
    out.array[:] = a.flatten()


def fill(out, val):
    out.array.fill(val)


def compact(a, out, shape, strides, offset):
    out.array[:] = to_numpy(a, shape, strides, offset).flatten()


def ewise_setitem(a, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = a.array.reshape(shape)


def scalar_setitem(size, val, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = val


def ewise_add(a, b, out):
    out.array[:] = a.array + b.array


def scalar_add(a, val, out):
    out.array[:] = a.array + val


def ewise_mul(a, b, out):
    out.array[:] = a.array * b.array


def scalar_mul(a, val, out):
    out.array[:] = a.array * val


def ewise_div(a, b, out):
    out.array[:] = a.array / b.array


def scalar_div(a, val, out):
    out.array[:] = a.array / val


def scalar_power(a, val, out):
    out.array[:] = a.array**val


def ewise_maximum(a, b, out):
    out.array[:] = np.maximum(a.array, b.array)


def scalar_maximum(a, val, out):
    out.array[:] = np.maximum(a.array, val)


def ewise_eq(a, b, out):
    out.array[:] = (a.array == b.array).astype(np.float32)


def scalar_eq(a, val, out):
    out.array[:] = (a.array == val).astype(np.float32)


def ewise_ge(a, b, out):
    out.array[:] = (a.array >= b.array).astype(np.float32)


def scalar_ge(a, val, out):
    out.array[:] = (a.array >= val).astype(np.float32)


def ewise_log(a, out):
    out.array[:] = np.log(a.array)


def ewise_exp(a, out):
    out.array[:] = np.exp(a.array)


def ewise_tanh(a, out):
    out.array[:] = np.tanh(a.array)


def ewise_sin(a, out):
    out.array[:] = np.sin(a.array)


def ewise_cos(a, out):
    out.array[:] = np.cos(a.array)


def arange(out, n):
    """Fill output with [0, 1, 2, ..., n-1] as float32."""
    out.array[:n] = np.arange(n, dtype=np.float32)


def triu_mask(out, rows, cols, k, mask_val):
    """Build upper-triangular mask: out[i,j] = mask_val if j >= i+k, else 0."""
    mask = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            if j >= i + k:
                mask[i, j] = mask_val
    out.array[:rows*cols] = mask.flatten()


def embedding_lookup(weight, ids, out, num_ids, embedding_dim):
    """Lookup embeddings by float IDs (cast to int internally)."""
    weight_2d = weight.array.reshape(-1, embedding_dim)
    int_ids = ids.array[:num_ids].astype(np.int64)
    out.array[:num_ids * embedding_dim] = weight_2d[int_ids].flatten()


def matmul(a, b, out, m, n, p):
    out.array[:] = (a.array.reshape(m, n) @ b.array.reshape(n, p)).reshape(-1)


def reduce_max(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).max(axis=1)


def reduce_sum(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).sum(axis=1)
