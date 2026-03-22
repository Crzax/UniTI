"""The module.
"""
from typing import Any
from uniti.autograd import Tensor
from uniti import ops
import uniti.init as init
from functools import reduce


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

         
        self.weight = Parameter(init.kaiming_uniform(in_features,out_features,device=device,dtype=dtype,requires_grad=True))
        self.bias = Parameter(init.kaiming_uniform(out_features,1,device=device,dtype=dtype,requires_grad=True).transpose()) if bias else None
         

    def forward(self, X: Tensor) -> Tensor:
         
        input_shape = X.shape
        
        batch_dim = 1
        for s in input_shape[:-1]:
            batch_dim *= s
        
        X_flat = X.reshape((batch_dim, self.in_features))
        
        out_flat = X_flat.matmul(self.weight)
        
        if self.bias:
            out_flat = out_flat + self.bias.broadcast_to((batch_dim, self.out_features))
        
        target_shape = input_shape[:-1] + (self.out_features,)
        return out_flat.reshape(target_shape)
         


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
         
        total_num = reduce(lambda a,b: a * b, X.shape)//X.shape[0]
        return X.reshape((X.shape[0],total_num))
         


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
         
        return ops.relu(x)
         

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
         
        for module in self.modules:
          x = module(x)
        return x
         


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
         
        y_one_hot = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
        x = logits * y_one_hot
        return ops.summation(ops.logsumexp(logits,(1,))-ops.summation(x,axes=(1,)))/x.shape[0]
         


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
         
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, )
        self.running_var = init.ones(dim, device=device, dtype=dtype, )
         

    def forward(self, x: Tensor) -> Tensor:
         
        if self.training:
          n, k = x.shape
          mean = ops.summation(x,axes=(0,))/n
          var = ((x-mean.reshape((1,k)).broadcast_to((n,k)))**2).sum(axes=(0,))/n
          self.running_mean = (1-self.momentum)*self.running_mean +self.momentum*mean.data
          self.running_var = (1-self.momentum)*self.running_var +self.momentum*var.data
          norm = (x-mean.reshape((1,k)).broadcast_to((n,k))) / (var.reshape((1,k)).broadcast_to((n,k))+self.eps)**0.5
          return self.weight.reshape((1,k)).broadcast_to((n,k)) * norm + self.bias.reshape((1,k)).broadcast_to((n,k))
        else:
          n, k = x.shape
          norm = (x-self.running_mean.reshape((1,k)).broadcast_to((n,k))) / (self.running_var.reshape((1,k)).broadcast_to((n,k))+self.eps)**0.5
          return self.weight.reshape((1,k)).broadcast_to((n,k)) * norm + self.bias.reshape((1,k)).broadcast_to((n,k))
         

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
         
        self.w = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.b = Parameter(init.zeros(dim, device=device, dtype=dtype,  requires_grad=True))
         

    def forward(self, x: Tensor) -> Tensor:
         
        input_shape = x.shape
        feature_dim = input_shape[-1]
        batch_dim = 1
        for s in input_shape[:-1]:
            batch_dim *= s
        x_flat = x.reshape((batch_dim, feature_dim))
        n, k = batch_dim, feature_dim
        mean = (x_flat.sum(axes=(1,)) / k).reshape((n, 1)).broadcast_to((n, k))
        var = (((x_flat - mean) ** 2).sum(axes=(1,)) / k).reshape((n, 1)).broadcast_to((n, k))
        deno = (var + self.eps) ** 0.5
        norm = (x_flat - mean) / deno
        w_broad = self.w.reshape((1, k)).broadcast_to((n, k))
        b_broad = self.b.reshape((1, k)).broadcast_to((n, k))
        out = w_broad * norm + b_broad
        return out.reshape(input_shape)
         


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
         
        if not self.training:
          return x
        msk = init.randb(*x.shape, p=1-self.p, device = x.device, dtype=x.dtype)
        return x * msk / (1-self.p)
         


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
         
        return x + self.fn(x)
         
