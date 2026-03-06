from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
         
        max_z = Z.max(axis=1, keepdims=True)
        log_sum_exp = array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=1, keepdims=True)) + max_z
        return Z - array_api.broadcast_to(log_sum_exp, Z.shape)        
         

    def gradient(self, out_grad: Tensor, node: Tensor):
         
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(axis=1,keepdims=True)
        exp_z = exp(z-max_z)

        before_broad_shape = list(z.shape)
        before_broad_shape[1] = 1
        softmax_z = exp_z / summation(exp_z, axes=1).reshape(tuple(before_broad_shape)).broadcast_to(z.shape)
        out_grad_sum = summation(out_grad, axes=1).reshape((z.shape[0],1)).broadcast_to(z.shape)
        return (out_grad + (-softmax_z) * out_grad_sum,)
         


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
         
        reduce_axes = self.axes
        new_shape = list(Z.shape)
        if reduce_axes is None:
          new_shape = [1]*len(Z.shape)  
        elif isinstance(reduce_axes, int):
          new_shape[reduce_axes] = 1
        else:
           for axis in reduce_axes:
              new_shape[axis] = 1
        new_shape = tuple(new_shape)
        maxz = Z.max(axis=reduce_axes, keepdims=True).reshape(new_shape).broadcast_to(Z.shape)
        maxzplus = Z.max(axis=reduce_axes)
        return array_api.log(array_api.sum(array_api.exp(Z-maxz),axis=reduce_axes)) + maxzplus
         

    def gradient(self, out_grad: Tensor, node: Tensor):
         
        z = node.inputs[0]
       
        reduce_shape = list(z.shape)
        if self.axes is None:
            axes_iter = range(len(reduce_shape))
        elif isinstance(self.axes, int):
            axes_iter = (self.axes,)
        else:
            axes_iter = self.axes
            
        for axis in axes_iter:
            reduce_shape[axis] = 1
        
        reduce_shape = tuple(reduce_shape)
        
        max_z = z.realize_cached_data().max(axis=self.axes, keepdims=True).reshape(reduce_shape).broadcast_to(z.shape)
        exp_z = exp(z-max_z)
        partial_z = exp_z / summation(exp_z, axes=self.axes).reshape(reduce_shape).broadcast_to(z.shape)
        return (partial_z * out_grad.reshape(tuple(reduce_shape)).broadcast_to(z.shape),)
         


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)