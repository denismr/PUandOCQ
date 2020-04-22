import numpy as np
import math
from OCQ.PositiveAutoSample import PAS
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

class ODIn:
  def __init__(self, tr_df, scorer, nfolds=10):
    scores = PAS(scorer, tr_df, n_splits=nfolds)
    scores.sort()
    # percentiles = np.arange(0, 101, 10)
    percentiles = [0.1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99.9]
    self.thresholds = np.percentile(scores, percentiles)
    self.overflow_limit = EstimateOverflowLimit(scores, self.thresholds)
    self.histogram = CreateHistogram(scores, self.thresholds)
    self.scorer = scorer(tr_df)

  def __call__(self, test_df):
    scores = self.scorer(test_df)
    histogram = CreateHistogram(scores, self.thresholds)
    return FindP(self.histogram, histogram, self.overflow_limit)
