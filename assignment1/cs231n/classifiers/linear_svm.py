import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shap 
  e (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin  
        dW[:, j] += X[i, :].T
        dW[:, y[i]] -= X[i, :].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # Average gradients as well
  dW /= num_train

  # Add regularization to the gradient
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  scores = X.dot(W)
  correct_score = np.reshape(scores[np.arange(scores.shape[0]), y], (scores.shape[0],-1)) # correct score: N x 1
  correct_score = np.ones(scores.shape) * correct_score
  L = scores - correct_score + 1 # N x C
  
  L[L < 0] = 0
  L[np.arange(scores.shape[0]), y] = 0
  
  loss = np.sum(L) / num_train
  loss += reg * np.sum(W * W)

  L_count = np.copy(L) # N X C
  L_count[L_count > 0] = 1
  L_count[np.arange(num_train), y] = -np.sum(L_count, axis=1)
  dW = np.dot(X.T, L_count)
  
  # Average gradients as well
  dW /= num_train

  # Add regularization to the gradient
  dW += reg * W

  return loss, dW
