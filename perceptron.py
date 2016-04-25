# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:29:49 2016

@author: hari
"""

import numpy as np

class Perceptron:
  """
    Parameters:
      learning_rate - float
      labels - {array-like}, shape = [2, 1]
    
    Attributes:
      weights_ - {array}, shape = [num_features, 1]
  """  
  
  def __init__(self, learning_rate = 0.01, labels = [-1, 1]):
    self.learning_rate = learning_rate
    self.labels = labels
    self.weights_ = None
  
  def train(self, XX, yy, n_iter = 100, allowable_erros = 0, 
            reinitialize_weights = False):
    if self.weights_ == None or reinitialize_weights == True:
      self.weights_ = np.zeros(1 + XX.shape[1])
        
    error_counts = []
    for _ in range(n_iter):
      errors = 0
      for (X, y) in zip(XX, yy):
        error = (np.where(y == self.labels[0], -1, 1) - 
                 self.predict(X))
        if error != 0:
          errors += 1
          update_factor = self.learning_rate * error
          self.weights_[1:] += update_factor * X
          self.weights_[0] += update_factor
      error_counts.append(errors)
      if errors == 0:
        break
    
    return error_counts
  
  def predict(self, X):
    if self.weights_ == None:
      return(None)
    else:
      return(np.where(np.dot(X, self.weights_[1:]) + self.weights_[0] < 0, 
                        -1, 
                        1))