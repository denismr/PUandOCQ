# Wrapper for KM1 and KM2

import pandas as pd
from KernelMPE import wrapper

def Both(data, features, labeled_info, nlimit=2000):
  if len(data) > nlimit:
    data = data[:nlimit]
    
  labeled = [x for x in data if x[labeled_info] == 1]
  labeled_np = pd.DataFrame(labeled)[features].values

  unlabeled = [x for x in data if x[labeled_info] == 0]
  unlabeled_np = pd.DataFrame(unlabeled)[features].values

  km1, km2 = wrapper(unlabeled_np, labeled_np)
  return km1, km2, len(labeled), len(unlabeled)

def KM1(data, features, labeled_info, nlimit=2000):
  ''' KM1
    Args:
      data (list): list of observation points (dictionaries with [feature]:value).
      features (list): list of which features should be considered.
      labeled_info (str): name of the feature that indicates whether obseration point is labeled or not (1 or 0, respectively).
      nlimit (int): size limite for the sample.
    
    Returns:
      (pred_c, pred_alpha): predicted c and predicted p (p is the proportion of positive observation points within the UNLABELED portion of the data).
  '''
  pred_alpha, _, nlbl, nunlbl = Both(data, features, labeled_info, nlimit=nlimit)
  pred_c = max(0, min(1, nlbl / (pred_alpha * nunlbl + nlbl)))
  return pred_c, pred_alpha

def KM2(data, features, labeled_info, nlimit=2000):
  ''' KM1
    Args:
      data (list): list of observation points (dictionaries with [feature]:value).
      features (list): list of which features should be considered.
      labeled_info (str): name of the feature that indicates whether obseration point is labeled or not (1 or 0, respectively).
      nlimit (int): size limite for the sample.
    
    Returns:
      (pred_c, pred_alpha): predicted c and predicted p (p is the proportion of positive observation points within the UNLABELED portion of the data).
  '''
  _, pred_alpha, nlbl, nunlbl = Both(data, features, labeled_info, nlimit=nlimit)
  pred_c = max(0, min(1, nlbl / (pred_alpha * nunlbl + nlbl)))
  return pred_c, pred_alpha