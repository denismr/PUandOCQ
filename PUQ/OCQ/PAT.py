import numpy as np
from OCQ.PositiveAutoSample import PAS

class PAT:
  def __init__(self, tr_df, scorer, nfolds=10, lbound=25, rbound=75, samples=None):
    scores = PAS(scorer, tr_df, n_splits=nfolds)
    scores.sort()
    if samples is None:
      samples = rbound - lbound
    percentiles = np.arange(lbound, rbound + 1, (rbound - lbound) / samples)
    self.quantiles = percentiles / 100
    self.thresholds = np.percentile(scores, percentiles)
    self.scorer = scorer(tr_df)

  def __call__(self, test_df):
    t_scores = self.scorer(test_df)
    t_scores.sort()

    n = len(t_scores)
    alphas = []
    for qnt, thr in zip(self.quantiles, self.thresholds):
      n_neg = float(np.searchsorted(t_scores, thr, side="right"))
      n_pos = n - n_neg
      alphas.append(min(1, (n_pos / n) / (1 - qnt)))
    return np.median(alphas)
  
