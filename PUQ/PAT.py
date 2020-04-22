# Passive Aggressive Threshold (PAT)

import pandas as pd
import numpy as np
from KFold import KFold
import math
from OCQ.PAT import PAT as OCQ_PAT
from timeit import default_timer as timer

def PAT(data, features, labeled_info, scorer, nfolds=10):
  ''' Passive Aggressive Threshold (PAT)
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
  labeled_df = pd.DataFrame(labeled)[features]
  pat = OCQ_PAT(labeled_df, scorer, nfolds)
  
  unlabeled = [x for x in data if x[labeled_info] == 0]
  unlabeled_df = pd.DataFrame(unlabeled)[features]

  alpha = pat(unlabeled_df)
  c = max(0, min(1, len(labeled_df) / (alpha * len(unlabeled_df) + len(labeled_df))))
  return c, alpha


# The following version of PAT was modified to return the time consumed to quantify (disregarding training).
# It is used inside ExperimentTimePAT.py and responsible for the additional column in the paper.

def PAT_ActualTime(data, features, labeled_info, scorer, nfolds=10):
  labeled = [x for x in data if x[labeled_info] == 1]
  labeled_df = pd.DataFrame(labeled)[features]
  pat = OCQ_PAT(labeled_df, scorer, nfolds)

  unlabeled = [x for x in data if x[labeled_info] == 0]
  unlabeled_df = pd.DataFrame(unlabeled)[features]

  tm_start = timer()
  alpha = pat(unlabeled_df)
  tm_end = timer()
  c = max(0, min(1, len(labeled_df) / (alpha * len(unlabeled_df) + len(labeled_df))))
  return c, alpha, (tm_end - tm_start)

