"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import uniti as uti

import uniti.nn as nn
from apps.models import *
import time
device = uti.cpu()

def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.open(image_filesname, 'rb') as imgs:
      magic = struct.unpack('>I', imgs.read(4))[0]
      assert(magic==2051)
      num_imgs = struct.unpack('>I', imgs.read(4))[0]
      rows = struct.unpack('>I', imgs.read(4))[0]
      columns = struct.unpack('>I', imgs.read(4))[0]
      img_data = imgs.read()
      X = np.frombuffer(img_data, dtype=np.uint8).reshape((num_imgs, rows * columns))
      X = np.astype(X, np.float32) / 255.0
    
    with gzip.open(label_filename, 'rb') as lbs:
      magic = struct.unpack('>I', lbs.read(4))[0]
      assert(magic==2049)
      num = struct.unpack('>I', lbs.read(4))[0]
    
      lbs_data = lbs.read()
      y = np.frombuffer(lbs_data, dtype=np.uint8)
    return (X, y)
     


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (uti.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (uti.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (uti.Tensor[np.float32])
    """
    return uti.summation(uti.log(uti.summation(uti.exp(Z), axes=(1,)))- (Z*y_one_hot).sum(axes=(1,)))/y_one_hot.shape[0]

     


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (uti.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (uti.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: uti.Tensor[np.float32]
            W2: uti.Tensor[np.float32]
    """
    iter_nums = (y.size + batch - 1) // batch
    for i in range(iter_nums):
      X_batch = uti.Tensor(X[i*batch:min((i+1)*batch, y.size),:])
      y_batch = y[i*batch:min((i+1)*batch, y.size)]
      if (i == iter_nums - 1): batch = y.size - i * batch
      y_one_hot = np.zeros((batch,y.max()+1))
      y_one_hot[np.arange(batch),y_batch]=1
      Z = uti.matmul(uti.relu(uti.matmul(X_batch, W1)), W2)
      loss = softmax_loss(Z,y_one_hot)
      loss.backward()
      W1 = uti.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
      W2 = uti.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
    return W1, W2
     

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
     
    acc_all, loss_all = 0.0, 0.0
    if opt is None:
      model.eval()
      for X, y in dataloader:
        logits = model(X)
        loss = loss_fn(logits, y)
        loss_all += loss.numpy()*y.shape[0]
        acc_all += np.sum(logits.numpy().argmax(axis=1) == y.numpy())
    else:
      model.train()
      for X, y in dataloader:
        logits = model(X)
        loss = loss_fn(logits, y)
        loss_all += loss.numpy()*y.shape[0]
        acc_all += np.sum(logits.numpy().argmax(axis=1) == y.numpy())
        opt.reset_grad()
        loss.backward()
        opt.step()
    sample_nums = len(dataloader.dataset)
    return acc_all / sample_nums, loss_all/sample_nums
     


def train_cifar10(model, dataloader, n_epochs=1, optimizer=uti.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    avg_acc, avg_loss = 0.0, 0.0
    opt = optimizer(params=model.parameters(), lr=lr,weight_decay=weight_decay)
    for _ in range(n_epochs):
      avg_acc, avg_loss = epoch_general_cifar10(dataloader=dataloader,model=model,loss_fn=loss_fn(),opt=opt)
    return avg_acc, avg_loss
     


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
     
    return epoch_general_cifar10(dataloader=dataloader,model=model,loss_fn=loss_fn())
     


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    
     
    nbatch, batch_size = data.shape
    avg_acc, avg_loss = 0.0, 0.0
    total_samples = 0
    
    if opt:
        model.train()
    else:
        model.eval()
    import gc
    h = None
    for i in range(0, nbatch - 1, seq_len):
        X, y = uti.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        
        if h is not None:
             if isinstance(h, tuple):
                 h = tuple(x.detach() for x in h)
             else:
                 h = h.detach()

        logits, h = model(X, h)
        loss = loss_fn(logits, y)
        
        if opt:
            opt.reset_grad()
            loss.backward()
            if clip is not None:
                opt.clip_grad_norm(clip)
            opt.step()
        
        batch_cnt = y.shape[0]
        total_samples += batch_cnt

        avg_loss += loss.numpy() * batch_cnt
        avg_acc += np.sum(logits.numpy().argmax(axis=1) == y.numpy())
        del logits, loss, X, y
        gc.collect()
    del h
    gc.collect()
    return avg_acc/total_samples, avg_loss/total_samples
     


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=uti.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
     
    avg_acc, avg_loss = 0.0, 0.0
    opt = optimizer(params=model.parameters(), lr=lr,weight_decay=weight_decay)
    for _ in range(n_epochs):
      avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn(), opt, clip, device, dtype)
    return avg_acc, avg_loss
     

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
     
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn(), device=device, dtype=dtype)
    return avg_acc, avg_loss
     

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = uti.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
