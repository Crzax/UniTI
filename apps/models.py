import sys
sys.path.append('./python')
import uniti as uti
import uniti.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(uti.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
         
        self.path = nn.Sequential(nn.ConvBN(3,16,7,4,device=device,dtype=dtype),
                                  nn.ConvBN(16,32,3,2,device=device,dtype=dtype),
                                  nn.Residual(nn.Sequential(nn.ConvBN(32,32,3,1,device=device,dtype=dtype),
                                  nn.ConvBN(32,32,3,1,device=device,dtype=dtype))),
                                  nn.ConvBN(32,64,3,2,device=device,dtype=dtype),
                                  nn.ConvBN(64,128,3,2,device=device,dtype=dtype),
                                  nn.Residual(nn.Sequential(nn.ConvBN(128,128,3,1,device=device,dtype=dtype),
                                  nn.ConvBN(128,128,3,1,device=device,dtype=dtype)))  ,
                                  nn.Flatten(),
                                  nn.Linear(128,128,device=device,dtype=dtype),
                                  nn.ReLU(),
                                  nn.Linear(128,10,device=device,dtype=dtype),
                                  )
         

    def forward(self, x):
         
        return self.path(x)
         


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
         
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emd = nn.Embedding(output_size, embedding_size, device, dtype)
        if seq_model == "rnn":
          self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
          self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        elif seq_model == 'transformer':
          self.seq_model = nn.Transformer(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        else:
          raise ValueError('unsupported seq_model. Only support rnn, lstm and transformer!')
        if seq_model == 'transformer':
          self.linear = nn.Linear(embedding_size, output_size, device=device, dtype=dtype)
        else:
          self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
         

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
         
        sl, bs = x.shape
        out = self.emd(x)
        out, h = self.seq_model(out, h)
        out = self.linear(out.reshape((sl*bs, out.shape[-1])))
        return out.reshape((sl*bs, self.output_size)), h
         


if __name__ == "__main__":
    model = ResNet9()
    x = uti.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = uti.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = uti.data.DataLoader(cifar10_train_dataset, 128, uti.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
