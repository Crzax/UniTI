"""The module.
"""
from typing import List, Callable, Any
from uniti.autograd import Tensor
from uniti import ops
import uniti.init as init
import math
from .nn_basic import Parameter, Module, BatchNorm2d, ReLU


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

         
        fan_in = in_channels * (kernel_size ** 2)
        fan_out = out_channels * (kernel_size ** 2)
        self.weight = Parameter(init.kaiming_uniform(fan_in,fan_out,shape=(kernel_size,kernel_size,in_channels,out_channels),device=device,dtype=dtype,requires_grad=True))
        interval = 1/math.sqrt(in_channels*kernel_size**2)
        self.padding = (kernel_size-1)//2
        self.bias = Parameter(init.rand(out_channels,low=-interval, high=interval, device=device,dtype=dtype,requires_grad=True)) if bias else None
         

    def forward(self, x: Tensor) -> Tensor:
         
        x_t = x.transpose((1,2)).transpose((2,3))
        out = ops.conv(x_t,self.weight,self.stride,self.padding)
        if self.bias is not None:
          out += self.bias.reshape((1,1,1,self.out_channels)).broadcast_to(out.shape)
        out = out.transpose((2,3)).transpose((1,2))
        return out
         

class ConvBN(Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
    super().__init__()
    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]
    if isinstance(stride, tuple):
        stride = stride[0]
    self.conv = Conv(in_channels, out_channels, kernel_size, stride, bias, device, dtype)
    self.bn = BatchNorm2d(dim=out_channels, device=device)
    self.relu = ReLU()
  
  def forward(self, x: Tensor) -> Tensor:
    x = self.conv(x)
    x = self.bn(x)
    return self.relu(x)