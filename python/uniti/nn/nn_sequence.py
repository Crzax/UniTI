"""The module.
"""
from typing import List
from uniti.autograd import Tensor
from uniti import ops
import uniti.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
         
        return (ops.exp(-x)+1)**-1
         

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
         
        self.device = device
        self.dtype = dtype
        if nonlinearity == "relu":
          self.nonlinearity = ops.ReLU()
        elif nonlinearity == "tanh":
          self.nonlinearity = ops.Tanh()
        else:
          raise ValueError("unsupported nonlinearity function. Only support ReLU and Tanh.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        interval = 1/hidden_size**0.5
        self.W_ih = Parameter(init.rand(*(input_size,hidden_size),low=-interval,high=interval,device=device,dtype=dtype))
        self.W_hh = Parameter(init.rand(*(hidden_size,hidden_size),low=-interval,high=interval,device=device,dtype=dtype))
        if bias:
          self.bias_ih = Parameter(init.rand(*(hidden_size,),low=-interval,high=interval,device=device,dtype=dtype))
          self.bias_hh = Parameter(init.rand(*(hidden_size,),low=-interval,high=interval,device=device,dtype=dtype))
        else:
          self.bias_ih = None
          self.bias_hh = None
         

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
         
        out = X @ self.W_ih

        batch_size, _ = X.shape
        if self.bias_ih is not None:
          b_ih = self.bias_ih.reshape((1, self.hidden_size))
          if batch_size > 1:
            b_ih = b_ih.broadcast_to((batch_size, self.hidden_size))
          out += b_ih
        if h is None:
          h = init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        
        out += h @ self.W_hh
        if self.bias_hh is not None:
          b_hh = self.bias_hh.reshape((1, self.hidden_size))
          if batch_size > 1:
            b_hh = b_hh.broadcast_to((batch_size, self.hidden_size))
          out += b_hh
        return self.nonlinearity(out)
         


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
         
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)] + [RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype) for _ in range(num_layers-1)]
         

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
         
        _, bs, _ = X.shape
        if h0 is None:
          h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
        h_status = list(ops.split(h0, 0))
        X_list = ops.split(X, 0)
        last_layer_out = []
        for t in range(len(X_list)):
          current_layer_in = X_list[t]
          for i in range(self.num_layers):
            h_status[i] = self.rnn_cells[i](current_layer_in, h_status[i])
            current_layer_in = h_status[i]
          last_layer_out.append(current_layer_in)
        
        return ops.stack(last_layer_out, 0), ops.stack(h_status, 0)
         


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
         
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        k = 1/hidden_size**0.5
        self.W_ih = Parameter(init.rand(*(input_size, 4*hidden_size),low=-k,high=k,device=device,dtype=dtype))
        self.W_hh = Parameter(init.rand(*(hidden_size, 4*hidden_size),low=-k,high=k,device=device,dtype=dtype))
        if bias:
          self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-k, high=k, device=device, dtype=dtype))
          self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-k, high=k, device=device, dtype=dtype))
        else:
          self.bias_ih = None
          self.bias_hh = None

         


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
         
        bs, _ = X.shape
        if h is None:
          h = init.zeros(*(bs, self.hidden_size),device=self.device,dtype=self.dtype)
          c = init.zeros(*(bs, self.hidden_size),device=self.device,dtype=self.dtype)
        else:
          h, c = h
        ifgo = X @ self.W_ih + h @ self.W_hh
        
        if self.bias_ih is not None:
          total_bias = (self.bias_ih + self.bias_hh).reshape((1, 4 * self.hidden_size))
          if bs > 1:
              total_bias = total_bias.broadcast_to((bs, 4 * self.hidden_size))
          ifgo = ifgo + total_bias
        ifgo = ifgo.reshape((bs, 4, self.hidden_size)) 
        gates = ops.split(ifgo, 1)
        i = gates[0].reshape((bs, self.hidden_size))
        f = gates[1].reshape((bs, self.hidden_size))
        g = gates[2].reshape((bs, self.hidden_size))
        o = gates[3].reshape((bs, self.hidden_size))
        sigmoid = Sigmoid()
        i,f,g,o = sigmoid(i),sigmoid(f),ops.tanh(g),sigmoid(o)
        c_out =f*c+i*g
        h_out = o*ops.tanh(c_out)
        return h_out, c_out
         


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
         
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)]+\
                [LSTMCell(hidden_size, hidden_size, bias, device, dtype) for _ in range(num_layers-1)]
         

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
         
        sl, bs, _ = X.shape
        if h is None:
          h = init.zeros(self.num_layers, bs, self.hidden_size, device =self.device, dtype = self.dtype)
          c = init.zeros(self.num_layers, bs, self.hidden_size, device =self.device, dtype = self.dtype)
        else:
          h, c = h
        h = list(ops.split(h, 0))
        c = list(ops.split(c, 0))
        output = []
        X_list = ops.split(X, 0)
        for t in range(sl):
          hin = X_list[t]
          for k in range(self.num_layers):
            h[k], c[k] = self.lstm_cells[k](hin, (h[k],c[k]))
            hin = h[k]
          output.append(hin)
        return ops.stack(output,0), (ops.stack(h,0), ops.stack(c,0))
         

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
         
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0, std=1, device=device,dtype=dtype))
         

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
         
        sl, bs = x.shape
        x_one_hot = init.one_hot(self.num_embeddings, x, device=self.device, dtype=self.dtype)
        return (x_one_hot.reshape((sl*bs, self.num_embeddings)) @ self.weight).reshape((sl, bs, self.embedding_dim))
         