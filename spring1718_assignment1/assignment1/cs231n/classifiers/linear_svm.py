import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i] 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  num_train = X.shape[0]
  dW = np.zeros(W.shape) # initialize the gradient as zero
 
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

 # 2018/9/23
  Scores = np.matmul(X,W)
  correct_class_score = Scores[np.arange(len(Scores)),y].reshape(-1,1)
  Scores = np.where(Scores-correct_class_score+1 > 0,Scores-correct_class_score+1,0)
  Scores[np.arange(len(Scores)),y] = 0
  loss = np.sum(Scores) / num_train + reg*np.sum(W*W)

  #2018/10/13 stage my forwarding computation.
  '''
  Scores = np.matmul(X,W)
  correct_class_score = Scores[np.arange(len(Scores)),y].reshape(-1,1)
  Margins = np.where(Scores - correct_class_score + 1 > 0, Scores - correct_class_score + 1, 0)
  data_loss = np.sum(Margins)/num_train
  reg_loss = reg * np.sum(W*W)
  loss = data_loss + reg_loss
  '''
    

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
    
  #2018/9/28
  """
  - X: numpy array (N,D)
  - Scores: numpy array (N,C)
  - W : numpy array (D,C) 
  - dW: numpy array (D,C)
  """
  Indexes = Scores > 0
  temp = np.sum(Indexes,axis = 1)
  Correct_Indexes = np.zeros(Indexes.shape)
  Correct_Indexes[np.arange(len(Correct_Indexes)),y] = temp
  dW = np.matmul(np.transpose(X),Indexes)- np.matmul(np.transpose(X),Correct_Indexes)
  dW = dW/num_train + 2*reg*W
  
   

  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
