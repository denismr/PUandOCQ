# Exhaustive Tree Induction fo c Estimation (ExTIcE)
# TODO: rewrite the code so that it handles all data with Pandas.

import math
import heapq
from numpy import median
from numpy import mean
from random import random
from KFold import KFold

def T(d):
  return len(d)

def L(d, labeled_info):
  return sum(map(lambda x: x[labeled_info], d))

def EvaluateSimple(d, delta, c, labeled_info):
  l = L(d, labeled_info)
  t = T(d)
  return l / t

def EvaluatePaper(d, delta, c, labeled_info):
  l = L(d, labeled_info)
  t = T(d)
  return l / t - math.sqrt((c * (1 - c) * (1 - delta)) / (delta * t))

def TIcE(data, features, labeled_info, nfolds=5, M=500, evaluation=EvaluatePaper):
  ''' Exhaustive Tree Induction fo c Estimation (ExTIcE)
    Args:
      data (list): list of observation points (dictionaries with [feature]:value).
      features (list): list of which features should be considered.
      labeled_info (str): name of the feature that indicates whether obseration point is labeled or not (1 or 0, respectively).
      nfolds (int): number of folds to average the final prediction.
      M (int): hard limit for branching.
      evaluation (callable): evaluation measure of branch quality.
    
    Returns:
      (pred_c, pred_alpha): predicted c and predicted p (p is the proportion of positive observation points within the UNLABELED portion of the data).
  '''
  pred_c = 0.5
  for _ in range(2):
    clist = []
    for est_data, tree_data in KFold(nfolds, data):
      delta = max(0.025, 1 / (1 + 0.004 * T(est_data)))
      cbest = L(est_data, labeled_info) / T(est_data)
      pq = [(-evaluation(tree_data, delta, pred_c, labeled_info), random(), tree_data, est_data, features[:])]
      limit = max(30, min(1000, math.floor(0.5 + 0.1 * min(T(est_data), T(tree_data)))))
      m = 0
      while m < M and len(pq) > 0:
        ev, _, St, Se, feat = heapq.heappop(pq)
        m += 1
        if ev >= 0 or T(St) < limit or T(Se) < limit:
          continue

        nev = evaluation(Se, delta, pred_c, labeled_info)
        cbest = max(cbest, nev)
        
        nfeat = []
        for f in feat:
          med = median([x[f] for x in St])
          left_St = [x for x in St if x[f] <= med]
          left_Se = [x for x in Se if x[f] <= med]
          right_St = [x for x in St if x[f] > med]
          right_Se = [x for x in Se if x[f] > med]
          if T(left_St) == 0 or T(right_St) == 0:
            continue
          nfeat.append(f)
          if T(left_St) > limit and T(left_Se) > limit:
            heapq.heappush(pq, (-evaluation(left_St, delta, pred_c, labeled_info), random(), left_St, left_Se, nfeat))
          if T(right_St) > limit and T(right_Se) > limit:
            heapq.heappush(pq, (-evaluation(right_St, delta, pred_c, labeled_info), random(), right_St, right_Se, nfeat))
      clist.append(cbest)
    pred_c = mean(clist)

  l = L(data, labeled_info)
  pred_alpha = max(0, min(1, (l / pred_c - l) / (len(data) - l)))
  return pred_c, pred_alpha