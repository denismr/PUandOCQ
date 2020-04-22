
# One Class Scorers
# This type of scorer creates scores based on only one class.

## You can either call an IMMEDIATE scorer, or create a PRE-TRAINED scorer
## Contrary to PRE-TRAINED scorers, in IMMEDIATE scorers, training is done again for every test data
## scorer(training) -> creates a persistent scorer
## scorer(training, test) -> creates an immediante scorer and immediately scores the test sample
## Only Mahalanobis does not have to be constructed first. All other functions must be first called to construt a scorer.

import numpy as np
import heapq
import math
from scipy.spatial.distance import mahalanobis
from scipy.stats import percentileofscore
from sklearn.svm import OneClassSVM as ocsvm
from sklearn.neighbors import LocalOutlierFactor
from sys import stderr
from collections import Counter
from sklearn.ensemble import IsolationForest

## Direct call
def Mahalanobis(training, test=None):
  V = training.cov()
  mu = np.array(training.mean(axis=0))
  try:
    VI = np.linalg.inv(V)
  except:
    print('warning: singular matrix', file=stderr)
    VI = np.linalg.pinv(V)

  def _test(test):
    dists = []
    for _, row in test.iterrows():
      lrow = np.array(row)
      d = -mahalanobis(lrow, mu, VI)
      if not math.isnan(d): dists.append(d)

    return dists
  
  return _test(test) if test is not None else _test

## Direct call
def SquaredMahalanobis(training, test=None):
  '''Squared Mahalanobis Distance. Use this if you are having problems with Mahalanobis distance (weird negative values).
  Args:
    training (pandas dataframe): training data.
    test (pandas dataframe): optional, data that must be scored.

  Returns:
    pre-trained scorer function (if test is None) or list of scores.
  '''
  V = training.cov()
  mu = np.array(training.mean(axis=0))
  try:
    VI = np.linalg.inv(V)
  except:
    print('warning: singular matrix', file=stderr)
    VI = np.linalg.pinv(V)

  def _test(test):
    dists = []
    for _, row in test.iterrows():
      lrow = np.array(row)
      diff = mu - lrow
      d = -np.matmul(np.matmul(diff, VI), diff)
      dists.append(d)

    return dists
  
  return _test(test) if test is not None else _test

def OneClassSVM(gamma="auto"):
  '''One-class SVM (OSVM) construtor.
  Args:
    gamma (str): gamma parameter of SVM.

  Returns:
    A scorer function that works exactly as the Mahalanobis function above.
  '''
  def f(training, test=None):
    '''Configured One-Class SVM.
    Args:
      training (pandas dataframe): training data.
      test (pandas dataframe): optional, data that must be scored.

    Returns:
      pre-trained scorer function (if test is None) or list of scores.
    '''
    svm = ocsvm(gamma=gamma)
    svm.fit(training)
    def _test(test):
      try:
        ret = [x for x in svm.decision_function(test) if not math.isnan(x)]
      except:
        print('Warning: dealing with exception in SVM DF')
        ret = []
        for it in range(len(test)):
          testi = test.iloc[[it]]
          try:
            dcf = svm.decision_function(testi)[0]
            if not math.isnan(dcf): ret.append(dcf)
          except:
            print('Warning: Culprit')
            print(testi)
      return ret
    return _test(test) if test is not None else _test
  return f

def OneClassSVM_raw(gamma="auto"):
  '''One-class SVM -- raw scores (constructor)
  Args:
    gamma (str): gamma parameter of SVM.

  Returns:
    A scorer function that works exactly as the Mahalanobis function above.
  '''
  def f(training, test=None):
    '''Configured One-Class SVM.
    Args:
      training (pandas dataframe): training data.
      test (pandas dataframe): optional, data that must be scored.

    Returns:
      pre-trained scorer function (if test is None) or list of scores.
    '''
    svm = ocsvm(gamma=gamma)
    svm.fit(training)
    def _test(test):
      return [x for x in svm.score_samples(test) if not math.isnan(x)]
    return _test(test) if test is not None else _test
  return f

def LOF(*vargs, **kargs):
  '''Local Outlier Factor (LOF) constructor.
  Args:
    *vargs: vargs for sklearn's LocalOutlierFactor.
    **kargs: kargs for sklearn's LocalOutlierFactor.

  Returns:
    A scorer function that works exactly as the Mahalanobis function above.
  '''
  def f(training, test=None):
    model = LocalOutlierFactor(*vargs, novelty=True, **kargs)
    model.fit(training)
    def _test(test):
      return [x for x in model.score_samples(test)]
    return _test(test) if test is not None else _test
  return f

def IF(*vargs, **kargs):
  '''IsolationForest (IF) constructor.
  Args:
    *vargs: vargs for sklearn's IsolationForest.
    **kargs: kargs for sklearn's IsolationForest.

  Returns:
    A scorer function that works exactly as the Mahalanobis function above.
  '''
  def f(training, test=None):
    model = IsolationForest(*vargs, **kargs)
    model.fit(training)
    def _test(test):
      return [x for x in model.score_samples(test)]
    return _test(test) if test is not None else _test
  return f