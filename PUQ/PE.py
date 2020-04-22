import pandas as pd
import numpy as np
import math
from functools import reduce
from KFold import KFold

def PE(data, features, labeled_info, nfolds=5):
  ''' PE
    Args:
      data (list): list of observation points (dictionaries with [feature]:value).
      features (list): list of which features should be considered.
      labeled_info (str): name of the feature that indicates whether obseration point is labeled or not (1 or 0, respectively).
      nfolds (int): number of folds for cross validation.
    
    Returns:
      (pred_c, pred_alpha): predicted c and predicted p (p is the proportion of positive observation points within the UNLABELED portion of the data).
  '''

  labeled = [x for x in data if x[labeled_info] == 1]
  unlabeled = [x for x in data if x[labeled_info] == 0]
  
  X = pd.DataFrame(data)[features].values
  xp = pd.DataFrame(labeled)[features].values
  xm = pd.DataFrame(unlabeled)[features].values

  n1, n2 = len(labeled), len(unlabeled)

  mm = np.matmul
  sqnorm = lambda x: x.dot(x)
  # med_dist = np.median([norm(x - y) for x in X for y in X])
  med_dist = np.sqrt(np.median([sqnorm(x - y) for x in X for y in X]))
  sigma_list = np.linspace(1/5, 5, 10) * med_dist

  lambda_list = np.logspace(-3, 1, 9)

  for (xp_tr, xp_te), (xm_tr, xm_te) in zip(KFold(nfolds, xp), KFold(nfolds, xm)):
    xp_tr, xp_te = np.array(xp_tr), np.array(xp_te)
    xm_tr, xm_te = np.array(xm_tr), np.array(xm_te)

    n1_tr, n1_te = len(xp_tr), len(xp_te)
    n2_tr, n2_te = len(xm_tr), len(xm_te)

    p1_tr = n1_tr / (n1_tr + n2_tr)
    p1_te = n1_te / (n1_te + n2_te)

    cv_scores = []

    for sigma in sigma_list:
      phi = lambda x: np.array([[math.exp(-sqnorm(x - xi)/(2 * sigma ** 2)) for xi in xp_tr]]).transpose()
      b = len(xp_tr)

      Phi1_tr = np.array([phi(x) for x in xp_tr])
      Phi1_te = np.array([phi(x) for x in xp_te])

      Phi2_tr = np.array([phi(x) for x in xm_tr])
      Phi2_te = np.array([phi(x) for x in xm_te])

      h_tr = np.mean(Phi1_tr, 0)
      h_te = np.mean(Phi1_te, 0)

      add = lambda x,y: x + y
      x = Phi1_tr[0]

      H_tr = p1_tr * reduce(add, (mm(x, x.transpose()) for x in Phi1_tr)) / n1_tr \
        + (1 - p1_tr) * reduce(add, (mm(x, x.transpose()) for x in Phi2_tr)) / n2_tr
      H_te = p1_te * reduce(add, (mm(x, x.transpose()) for x in Phi1_te)) / n1_te \
        + (1 - p1_te) * reduce(add, (mm(x, x.transpose()) for x in Phi2_te)) / n2_te

      for lamb in lambda_list:
        alpha = np.linalg.solve(H_tr + lamb * np.identity(b), h_tr)
        alpha_t = alpha.transpose()
        score = float(0.5 * mm(mm(alpha_t, H_te), alpha) - mm(alpha_t, h_te))
        cv_scores.append((score, lamb, sigma))
  
  _, lamb, sigma = min(cv_scores)
  phi = lambda x: np.array([[math.exp(-sqnorm(x - xi)/(2 * sigma ** 2)) for xi in xp]]).transpose()
  b = len(xp)

  Phi1 = np.array([phi(x) for x in xp])
  Phi2 = np.array([phi(x) for x in xm])
  p1 = n1 / (n1 + n2)
  h = np.mean(Phi1, 0)
  H = p1 * reduce(add, (mm(x, x.transpose()) for x in Phi1)) / n1 \
    + (1 - p1) * reduce(add, (mm(x, x.transpose()) for x in Phi2)) / n2
  alpha = np.linalg.solve(H + lamb * np.identity(b), h)
  alpha_t = alpha.transpose()
  prior = 1 / float((2 * mm(alpha_t, h) - mm(mm(alpha_t, H), alpha)))
  prior = min(1, max(prior, n1 / (n1 + n2)))

  c = max(0, min(1, len(labeled) / (prior * len(unlabeled) + len(labeled))))
  return c, prior