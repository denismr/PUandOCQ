# One Distribution Inside (ODIn)

import pandas as pd
import numpy as np
import math

from KFold import KFold
from random import shuffle

def CreateHistogram(scores, thresholds):
  scores.sort()
  hist = [0] * 12
  hist[0] = np.searchsorted(scores, thresholds[0], side="right")
  for i in range(1, 11):
    right = np.searchsorted(scores, thresholds[i], side="right")
    left = np.searchsorted(scores, thresholds[i - 1], side="right")
    hist[i] = right - left
  hist[11] = len(scores) - np.searchsorted(scores, thresholds[10], side="right")
  s = sum(hist)
  return [x / s for x in hist]

def Overflow(A, B, s):
  return sum((max(0, s * x[0] - x[1]) for x in zip(A, B)))

def BestFit(A, B, overflow_limit, eps):
  left = 0
  right = 1
  while abs(right - left) > eps:
    middle = (left + right) / 2
    check = Overflow(A, B, middle)
    if check <= middle * overflow_limit:
      left = middle
    else:
      right = middle
  return (left + right) / 2

def FindP(A, B, overflow_limit):
  p = BestFit(A, B, overflow_limit, 1e-5)
  return p - Overflow(A, B, p)

def EstimateOverflowLimit(scores, thresholds, iterations=100):
  w = math.floor(len(scores) / 3)
  s = 0
  s2 = 0

  for _ in range(iterations):
    shuffle(scores)
    dist_in = CreateHistogram(scores[:w], thresholds)
    dist_out = CreateHistogram(scores[w:2 * w], thresholds)
    v = Overflow(dist_in, dist_out, 1)
    s += v
    s2 += v * v

  mu = s / iterations
  sd = math.sqrt((s2 / iterations) - mu * mu)

  return mu + 3 * sd

def ODIn(data, features, labeled_info, scorer, nfolds=10):
  ''' One Distribution Inside (ODIn)
    Args:
      data (list): list of observation points (dictionaries with [feature]:value).
      features (list): list of which features should be considered.
      labeled_info (str): name of the feature that indicates whether obseration point is labeled or not (1 or 0, respectively).
      scorer (callable): One-class scorer to be used. See OCScorers.py for more information.
      nfolds (int): number of folds for cross validation to generate training scores.
    
    Returns:
      (pred_c, pred_alpha): predicted c and predicted p (p is the proportion of positive observation points within the UNLABELED portion of the data).
  '''
  labeled = [x for x in data if x[labeled_info] == 1]

  p_scores = []
  for tr, te in KFold(nfolds, labeled):
    tr_df = pd.DataFrame(tr)[features]
    te_df = pd.DataFrame(te)[features]
    p_scores += scorer(tr_df, te_df)

  labeled_df = pd.DataFrame(labeled)[features]
  unlabeled = [x for x in data if x[labeled_info] == 0]
  unlabeled_df = pd.DataFrame(unlabeled)[features]
  t_scores = scorer(labeled_df, unlabeled_df)

  percentiles = np.arange(0, 101, 10)
  thresholds = np.percentile(p_scores, percentiles)


  overflow_limit = EstimateOverflowLimit(p_scores, thresholds)

  p_histogram = CreateHistogram(p_scores, thresholds)
  t_histogram = CreateHistogram(t_scores, thresholds)

  p = FindP(p_histogram, t_histogram, overflow_limit)
  c = max(0, min(1, len(labeled) / (p * len(unlabeled) + len(labeled))))
  return c, p