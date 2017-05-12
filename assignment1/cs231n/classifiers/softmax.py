import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  for i in xrange(num_train):
        scores = np.exp(X[i].dot(W))
        scores /= np.sum(scores)
        loss += -np.log(scores[y[i]])
        
        L = np.copy(scores)
        L[y[i]] -= 1
        L = np.reshape(L, (1,-1))
        scale = np.reshape(X[i].T, (-1,1))
        dW += np.dot(scale, L)
  
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  dW  /= num_train
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  scores = np.exp(np.dot(X,W)) # N x C
  # scores -= np.max(scores, axis=1, keepdims=True)
  sum = np.sum(scores, axis=1)
  sum = np.reshape(sum, (-1,1)) # N x 1
  sum = np.ones(scores.shape) * sum # N x C
  scores = scores / sum
  loss = np.sum(-np.log(scores[np.arange(num_train), y]))

  L = np.copy(scores)
  L[np.arange(num_train), y] -= 1
  dW = np.dot(X.T, L)

  loss /= num_train
  loss +=  0.5 * reg * np.sum(W * W)
  
  dW  /= num_train
  dW += reg * W

  return loss, dW

