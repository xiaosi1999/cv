from builtins import range
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    for i in range(N):
      score = X[i].dot(W)  # (C,)
      exp_score = np.exp(score - np.max(score))  # (C,)
      loss += -np.log(exp_score[y[i]] / np.sum(exp_score))
      dexp_score = exp_score / np.sum(exp_score)
      dexp_score[y[i]] -= 1
      dscore = dexp_score
      dW += X[[i]].T.dot([dscore])  # 这里看似多余的括号是为了增加维度
    loss /= N
    dW /= N
    loss += reg * np.sum(W**2)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    scores = X.dot(W)  # (N, C)
    scores1 = scores - np.max(scores, axis=1, keepdims=True)  # (N, C)
    loss1 = -scores1[range(N), y] + np.log(np.sum(np.exp(scores1), axis=1))  # (N,)
    loss = np.sum(loss1) / N + reg * np.sum(W**2)
    dloss1 = np.ones((N, 1)) / N  # (N, 1)
    dscores1 = dloss1 * np.exp(scores1) / np.sum(np.exp(scores1), axis=1, keepdims=True)  # (N, C)
    dscores1[[range(N)], y] -= dloss1.T
    dscores = dscores1  # (N, C)
    dW = X.T.dot(dscores) + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
