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
  
  def train(self, XX, yy, n_iter = 100, allowable_errors = 0, 
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
  
  def net_output(self, X):
    return(np.dot(X, self.weights_[1:]) + self.weights_[0])
  
  def predict(self, X):
    if self.weights_ == None:
      return(None)
    else:
      return(np.where(self.net_output(X) < 0, -1, 1))

class OneVsAllPerceptron:
  """
    Parameters: 
      labels: list will all the possible labels
      perceptrons_: all the perceptrons (one for each label)
  """
  
  def __init__(self, labels):
    self.labels = labels
    self.perceptrons_ = [Perceptron() for l in self.labels]
  
  def train(self, XX, yy, n_iter = 100, allowable_errors = 0,
            reinitialize_weights = False):
    all_errors = []
    for i in range(len(self.labels)):
      yy_i = np.where(yy == self.labels[i], 1, -1)
      errors = self.perceptrons_[i].train(XX, yy_i, n_iter=n_iter, 
                                 allowable_errors=allowable_errors,
                                 reinitialize_weights=reinitialize_weights)
      all_errors.append(errors)
    return(all_errors)
      
  def predict(self, X):
    outputs = [p.net_output(X) for p in self.perceptrons_]
    return(self.labels[outputs.index(min(outputs))])