
# Algorithm to generate data for the Best Fixed Threshold (BFT) topline
# It outputs an array with 101 predictions for the proportion of positive observations in the sample.
# Each prediction is related to one threshold (percentiles from 0 to 100 of the training scores data)

import pandas as pd
import numpy as np
from KFold import KFold
import math

def BFT(data, features, labeled_info, scorer, nfolds=10):
  ''' Best Fixed Threshold
    Args:
      data (list): list of observation points (dictionaries with [feature]:value).
      features (list): list of which features should be considered.
      labeled_info (str): name of the feature that indicates whether obseration point is labeled or not (1 or 0, respectively).
      scorer (callable): One-class scorer to be used. See OCScorers.py for more information.
      nfolds (int): number of folds for cross validation to generate training scores.
    
    Returns:
      pred_alphas: 101 predicted p's, one for each percentile of training positive scores (p is the proportion of positive observation points within the UNLABELED portion of the data).
  '''
  labeled = [x for x in data if x[labeled_info] == 1]

  p_scores = []
  for tr, te in KFold(nfolds, labeled):
    tr_df = pd.DataFrame(tr)[features]
    te_df = pd.DataFrame(te)[features]
    p_scores += scorer(tr_df, te_df)
  p_scores.sort()

  labeled_df = pd.DataFrame(labeled)[features]
  unlabeled = [x for x in data if x[labeled_info] == 0]
  unlabeled_df = pd.DataFrame(unlabeled)[features]
  t_scores = scorer(labeled_df, unlabeled_df)
  t_scores.sort()

  percentiles = np.arange(0, 101, 1)

  thresholds = np.percentile(p_scores, percentiles)

  n = len(t_scores)
  alphas = []
  for thr in zip(thresholds):
    n_neg = float(np.searchsorted(t_scores, thr, side="right"))
    n_pos = n - n_neg
    alphas.append(n_pos / n)

  return np.array(alphas)