# Elkan's algorithm (EN)

import pandas as pd
import numpy as np
from random import shuffle
from KFold import KFold
from sklearn.svm import SVC

def EN(data, features, labeled_info, gamma="auto"):
  ''' Elkan's Algorithm (EN)
    Args:
      data (list): list of observation points (dictionaries with [feature]:value).
      features (list): list of which features should be considered.
      labeled_info (str): name of the feature that indicates whether obseration point is labeled or not (1 or 0, respectively).
      gamma (str): gamma parameter of SVM.
    
    Returns:
      (pred_c, pred_alpha): predicted c and predicted p (p is the proportion of positive observation points within the UNLABELED portion of the data).
  '''
  labeled = [x for x in data if x[labeled_info] == 1]
  test_data = pd.DataFrame(labeled)[features]
  training_data = pd.DataFrame(data)
  svc = SVC(probability=True, gamma=gamma)
  svc.fit(training_data[features], training_data[labeled_info])
  right_index = 0 if svc.classes_[0] == 1 else 1
  svc_probs = svc.predict_proba(test_data)
  pred_c = np.mean([x[right_index] for x in svc_probs])
  l = len(labeled)

  pred_alpha = max(0, min(1, (l / pred_c - l) / (len(data) - l)))
  return pred_c, pred_alpha


# Elkan with k-fold cross validation (not original Elkan)
def ENKF(data, features, labeled_info, k=5, gamma="auto"):
  ''' Elkan's Algorithm (EN) with k-fold cross validation (not original Elkan). Returned C is incorrect.
  '''
  labeled = [x for x in data if x[labeled_info] == 1]
  unlabeled = [x for x in data if x[labeled_info] != 1]

  all_probs = []
  all_alphas = []
  for tr, te in KFold(k, labeled):
    data_tr = unlabeled + tr
    shuffle(data_tr)
    test_data = pd.DataFrame(te)[features]
    training_data = pd.DataFrame(data_tr)
    svc = SVC(probability=True, gamma=gamma)
    svc.fit(training_data[features], training_data[labeled_info])
    right_index = 0 if svc.classes_[0] == 1 else 1
    svc_probs = svc.predict_proba(test_data)
    c_probs = [x[right_index] for x in svc_probs]
    all_probs += c_probs
    pred_c = np.mean(c_probs)
    l = len(tr)
    all_alphas += [max(0, min(1, (l / pred_c - l) / (len(data_tr) - l)))]

  pred_c = np.mean(all_probs)

  pred_alpha = np.mean(all_alphas)
  return pred_c, pred_alpha
