"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
         
        raise NotImplementedError()
         

    def gradient(self, out_grad, node):
         
        raise NotImplementedError()
         


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
         
        return a ** self.scalar
         

    def gradient(self, out_grad, node):
         
        return (out_grad * self.scalar * node.inputs[0]**(self.scalar-1),);
         


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
         
        return a/b
         

    def gradient(self, out_grad, node):
         
        lhs, rhs = node.inputs
        return out_grad/rhs, -out_grad*lhs/rhs**2
         


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
         
        return a/self.scalar
         

    def gradient(self, out_grad, node):
         
        return (out_grad / self.scalar,)
         


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
         
        new_axes = list(range(len(a.shape)))
        if (self.axes):
          i, j = self.axes[0], self.axes[1]
          new_axes[i], new_axes[j] = new_axes[j], new_axes[i] 
          return a.permute(new_axes)
        else:
          new_axes[-1], new_axes[-2] = new_axes[-2], new_axes[-1]
          return a.permute(new_axes)
        
         

    def gradient(self, out_grad, node):
         
        return transpose(out_grad, axes=self.axes)
         


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
         
        return a.compact().reshape(self.shape)
         

    def gradient(self, out_grad, node):
         
        return out_grad.reshape(node.inputs[0].shape)
         


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
         
        return array_api.broadcast_to(a, self.shape)
         

    def gradient(self, out_grad, node):
         
        ori_shape = node.inputs[0].shape
        dims = [i for i in range(len(self.shape))]
        for i, (ori_dim, now_dim) in enumerate(zip(reversed(ori_shape),reversed(self.shape))):
          if (ori_dim == now_dim):
            dims[len(self.shape)-1-i] = -1;
        dims = tuple(filter(lambda x: x>= 0, dims))
        if len(dims) == 0:
            return out_grad.reshape(ori_shape)
        return out_grad.sum(dims).reshape(ori_shape)
         


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
         
        if isinstance(self.axes, (list, tuple)) and len(self.axes) > 1:
          sorted_axes = sorted(self.axes, reverse=True)
          res = a
          for axis in sorted_axes:
              res = res.sum(axis=axis)
          return res
            
        return array_api.sum(a, axis=self.axes)
         

    def gradient(self, out_grad, node):
         
        input_shape = node.inputs[0].shape
        
        new_shape = list(input_shape)
        
        if self.axes is None:
            axes_to_expand = range(len(input_shape))
        elif isinstance(self.axes, int):
            axes_to_expand = [self.axes]
        else:
            axes_to_expand = self.axes
            
        for axis in axes_to_expand:
            new_shape[axis] = 1
            
        return out_grad.reshape(tuple(new_shape)).broadcast_to(input_shape)
         



def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
         
        return a @ b
         

    def gradient(self, out_grad, node):
         
        lhs, rhs = node.inputs
        gl, gr = matmul(out_grad, transpose(rhs)), matmul(transpose(lhs),out_grad)
        if (len(gl.shape)> len(lhs.shape)):
          gl = gl.sum(axes = tuple([i for i in range(len(gl.shape)-len(lhs.shape))]))
        if (len(gr.shape)> len(rhs.shape)):
          gr = gr.sum(axes = tuple([i for i in range(len(gr.shape)-len(rhs.shape))]))
        return gl, gr
         


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
         
        return -a;
         

    def gradient(self, out_grad, node):
         
        return -out_grad
         


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
         
        return array_api.log(a)
         

    def gradient(self, out_grad, node):
         
        return out_grad / node.inputs[0]
         


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
         
        return array_api.exp(a)
         

    def gradient(self, out_grad, node):
         
        return out_grad * exp(node.inputs[0])
         


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
         
        return array_api.maximum(a, 0)
         

    def gradient(self, out_grad, node):
         
        msk = node.realize_cached_data() > 0
        return out_grad * Tensor(msk, device = out_grad.device)
         


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
         
        return array_api.tanh(a)
         

    def gradient(self, out_grad, node):
         
        return out_grad * (-node**2 + 1)
         


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
         
        assert len(args) > 0, "Stack needs at least one array!"
        shape = args[0].shape
        for a in args:
          assert a.shape==shape, "All arrays need to be of the same size!"
        newshape = list(shape)
        n = len(args)
        newshape.insert(self.axis, n)
        out = array_api.empty(tuple(newshape), device = args[0].device)
        idxs = [slice(0, i) for i in newshape]
        for i in range(n):
          idxs[self.axis] = slice(i, i+1)
          out[tuple(idxs)] = args[i]
        return out
         

    def gradient(self, out_grad, node):
         
        return split(out_grad, self.axis)
         


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
         
        idxs = [slice(0, i) for i in A.shape]
        ans = []
        newshape = list(A.shape)
        split_dim_size = newshape.pop(self.axis)
        newshape = tuple(newshape)
        for i in range(split_dim_size):
          idxs[self.axis] = slice(i, i+1)
          ans.append(A[tuple(idxs)].compact().reshape(newshape))
        return tuple(ans)
         

    def gradient(self, out_grad, node):
         
        return stack(out_grad, self.axis)
         


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
         
        return a.flip(self.axes)
         

    def gradient(self, out_grad, node):
         
        return flip(out_grad, self.axes)
         


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
         
        new_shape = list(a.shape)
        for axis in self.axes:
          new_shape[axis] *= self.dilation + 1
        out = array_api.full(new_shape,0, device = a.device)
        idxs = tuple([slice(0, now_n, self.dilation+1 if now_n != ori_n else 1) for now_n, ori_n in zip(new_shape, a.shape)])
        out[idxs] = a
        return out
         

    def gradient(self, out_grad, node):
         
        return undilate(out_grad, self.axes, self.dilation)
         


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
         
        idxs = tuple([slice(0, elem, self.dilation+1 if i in self.axes else 1) for i, elem in enumerate(a.shape)])
        return a[idxs]
         

    def gradient(self, out_grad, node):
         
        return dilate(out_grad, self.axes, self.dilation)
         


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)

class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
         
        A_paded = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N,H,W,C_in = A_paded.shape
        K,_,_,C_out = B.shape
        Ns, Hs, Ws, Cs = A_paded.strides
    
        inner_dim = K * K * C_in
        out_h = (H-K)//self.stride+1
        out_w = (W-K)//self.stride+1
        Z = NDArray.make(shape = (N, out_h, out_w, K, K, C_in),
                  strides = (Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, Cs), 
                  device = A_paded.device, 
                  handle = A_paded._handle,
                  offset = A_paded._offset).compact().reshape((N*out_h*out_w,inner_dim))

        out = Z @ B.compact().reshape((inner_dim, C_out))
        out = out.reshape((N,out_h,out_w,C_out))
        return out
         

    def gradient(self, out_grad, node):
         
        X, W = node.inputs
        N, H_in, W_in, C_in = X.shape
        K, _, _, C_out = W.shape

        W_T_F = flip(W.transpose((2,3)), (0,1))
        if self.stride > 1: 
          out_d = dilate(out_grad, (1, 2), self.stride - 1) 
        else: 
          out_d = out_grad
        grad_X = conv(out_d, W_T_F, 1, K-1-self.padding)
        
        X_T = X.transpose((0,3))
        out_T_T = out_d.transpose((0,1)).transpose((1,2))
        grad_W = conv(X_T, out_T_T, 1, self.padding).transpose((0,1)).transpose((1,2))
        return (grad_X, grad_W)
         


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


