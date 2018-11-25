import numpy as np
from random import shuffle

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
    
  #2018/10/5 추가 #

  num_train = X.shape[0]
  num_classes = W.shape[1]
  tmp = 0.0
  correct_score =0.0

  for i in range(num_train):
      tmp = 0.0
      for j in range(num_classes):
          tmp += np.exp(W[:,j].dot(X[i]))
        
      correct_score = W[:,y[i]].dot(X[i])
      loss += -correct_score + np.log(tmp)
        
      for j in range(num_classes):
          dW[:,j] += X[i]*np.exp(W[:,j].dot(X[i]))/tmp
     
      #dW[:,y[i]] += -X[i] + X[i]*np.exp(W[:,y[i]].dot(X[i]))/tmp
      dW[:,y[i]] -= X[i]
    
  loss /= num_train 
  dW /= num_train
    
  loss += reg*np.sum(W*W)
  dW += 2*reg*W
  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  #2018/10/5
  '''
  Scores = np.matmul(X,W) # Scores : (N,C) 
  Exp_scores = np.exp(Scores)
  Sum_tmp = np.sum(Exp_scores,axis = 1) 
  Correct_scores = Scores[np.arange(len(Scores)),y]  
  loss = (np.sum(np.log(Sum_tmp)) - np.sum(Correct_scores))/num_train + reg*np.sum(W*W)
  
  tmp = np.zeros_like(Scores)
  tmp[np.arange(len(tmp)),y] = Exp_scores[np.arange(len(Exp_scores)),y]/Sum_tmp - 1
  Exp_scores /=Sum_tmp.reshape(-1,1)
  dW = (np.matmul(np.transpose(X),Exp_scores) + np.matmul(np.transpose(X),tmp))/num_train +2*reg*W
  '''
  #2018/10/13 modified : stage my computation
  
  Scores = np.matmul(X,W)
  margin = np.log(np.sum(np.exp(Scores),axis = 1)) - Scores[np.arange(len(Scores)),y]
  data_loss = np.sum(margin)/num_train
  reg_loss = reg*np.sum(W*W)
  loss = data_loss + reg_loss
  
  
  ddata_loss = 1
  dreg_loss = 1
  
  dmargin = ddata_loss * np.ones(len(margin)).reshape(-1,1)/num_train
  #print("dmargin's shape:",dmargin.shape)
  dScores = np.exp(Scores)/(np.sum(np.exp(Scores),axis=1).reshape(-1,1))
  dScores[np.arange(len(Scores)),y] -=  1
  #dScores *= dmargin
  dScores /= num_train

  dW = np.matmul(np.transpose(X),dScores)
  #dX = np.matmul(dScores,np.transpose(W))
  dW += 2*reg*W
  
  
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

