"""Optimization module"""
import uniti as uti
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
         
        for param in self.params:
          if self.weight_decay > 0:
            grad = param.grad.data + self.weight_decay * param.data
          else:
            grad = param.grad.data
          if param not in self.u:
            self.u[param] = uti.init.zeros(*param.shape, device=param.device, dtype=param.dtype).data
          self.u[param] = self.momentum*self.u[param] + (1-self.momentum)*grad
          param.data = param.data - self.lr * self.u[param]
         

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
         
        total_norm_sq = 0.0
        for p in self.params:
            if p.grad is not None:
                grad_np = p.grad.detach().numpy()
                total_norm_sq += np.sum(grad_np ** 2)
        
        total_norm = np.sqrt(total_norm_sq)
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1.0:
            for p in self.params:
                if p.grad is not None:
                    p.grad.data = p.grad.data * clip_coef
         


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
         
        self.t += 1
        for param in self.params:
          if self.weight_decay > 0:
            grad = param.grad.data + self.weight_decay * param.data
          else:
            grad = param.grad.data
          if param not in self.m:
            self.m[param] = uti.init.zeros(*param.shape,device=param.device,dtype=param.dtype).data
          if param not in self.v:
            self.v[param] = uti.init.zeros(*param.shape,device=param.device,dtype=param.dtype).data
          beta1, beta2 = self.beta1, self.beta2
          self.m[param] = beta1 * self.m[param] + (1-beta1) * grad
          self.v[param] = beta2 * self.v[param] + (1-beta2) * grad**2
          
          mbar=self.m[param]/(1-beta1**self.t)
          vbar=self.v[param]/(1-beta2**self.t)
          param.data = param.data - self.lr * mbar / (vbar**0.5 + self.eps)
         
