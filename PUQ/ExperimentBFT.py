# Generate data for the Best Fixed Threshold (BFT) topline
# This script outputs a CSV file with one column for each threhold.
# There are 101 thresholds (percentiles 0 to 100 of positive training scores).
# Another script is used to analyze the output and decides which threshold led to the best result.
# See Experiment.py to have a better grasp on the general idea of this file.

from timeit import default_timer as timer
import csv
import pandas as pd
import numpy as np
import heapq
import argparse
import sys

import math
from random import shuffle
from random import random
from KFold import KFold
from explist import Experiments as Exp
from BFT import BFT
import OCScorers
from tqdm import tqdm
from tqdm import trange

scorers_by_name = {
  "mah": OCScorers.Mahalanobis,
  "svm": OCScorers.OneClassSVM(),
  "svm_scl": OCScorers.OneClassSVM(gamma="scale"),
}

def eprint(*args, **kwargs):
    # print(*args, file=sys.stderr, **kwargs)
    pass

def RunExperiment(exp_name, scorer_name, niterations=5, max_sample_size=2000, max_labeled_size=500, nfolds=5):
  exp = Exp[exp_name]
  dataset_filename = exp["input"]
  output_filename = exp["output"]
  class_feature = exp["class_feature"]
  positive_label = exp["positive_label"]
  scorer = scorers_by_name[scorer_name]
  features = None
  negative_labels = None

  data_df = pd.read_csv(dataset_filename, index_col=False)

  if callable(exp["negative_labels"]):
    negative_labels = set(filter(exp["negative_labels"], set(data_df[class_feature])))
  elif isinstance(exp["negative_labels"], list):
    negative_labels = set(exp["negative_labels"])
  else:
    negative_labels = set([x for x in set(data_df[class_feature]) if x != positive_label])

  all_labels = set(list(negative_labels) + [positive_label])

  data_df = pd.DataFrame(data_df.loc[data_df[class_feature].map(lambda x: x in all_labels)])

  if callable(exp["features"]):
    features = list(filter(exp["features"], list(data_df)))
  elif isinstance(exp["features"], list):
    features = exp["features"]
  else:
    features = [x for x in list(data_df) if x != class_feature]


  labeled_info = 'dfjiweojgf'
  data = data_df.to_dict('registers')

  for_table = []
  for alpha in tqdm(list(np.linspace(0, 1, 11)), desc="alpha"):
    abs_errors = np.zeros(101)
    abs_errors_2 = np.zeros(101)
    errors = np.zeros(101)
    errors_2 = np.zeros(101)
    ms_per_example = []
    for it in trange(niterations):
      shuffle(data)
      eprint('Iteration %d#' % (it + 1))
      for fold_i, (unlabeled, all_labeled) in zip(trange(nfolds, desc="kfold"), KFold(nfolds, data)):
        eprint('  Fold #%d' % (fold_i + 1))
        for x in unlabeled: x[labeled_info] = 0
        for x in all_labeled: x[labeled_info] = 1
        labeled = [x for x in all_labeled if x[class_feature] == positive_label]
        positives = [x for x in unlabeled if x[class_feature] == positive_label]
        negatives = [x for x in unlabeled if x[class_feature] != positive_label]


        shuffle(labeled)
        shuffle(positives)
        shuffle(negatives)

        sample_size = min(len(positives), len(negatives), max_sample_size)
        npos = math.floor(alpha * sample_size)
        nneg = sample_size - npos
        nlab = min(len(labeled), max_labeled_size)

        sample = positives[:npos] + negatives[:nneg] + labeled[:nlab]
        shuffle(sample)

        actual_c = len(labeled) / (len(labeled) + npos)
        actual_alpha = npos / sample_size
        eprint('       Actual c: %6.2f |    Actual alpha: %6.2f' % (actual_c, actual_alpha))
        tm_start = timer()
        pred_alpha = BFT(sample, features, labeled_info, scorer)
        tm_end = timer()

        ms_per_example.append((tm_end - tm_start) * 1000 / len(sample))
        abs_errors += np.abs(actual_alpha - pred_alpha) / (niterations * nfolds)
        abs_errors_2 += np.abs(actual_alpha - pred_alpha) ** 2.0 / (niterations * nfolds)

        errors += (actual_alpha - pred_alpha) / (niterations * nfolds)
        errors_2 += (actual_alpha - pred_alpha) ** 2.0 / (niterations * nfolds)
    

    std_abs_errors = 100 * np.sqrt(abs_errors_2 - abs_errors ** 2)
    std_errors = 100 * np.sqrt(errors_2 - errors ** 2)
    abs_errors *= 100
    errors *= 100

    row = [100 * alpha, np.mean(ms_per_example), np.std(ms_per_example)] + list(abs_errors) + list(std_abs_errors) + list(errors) + list(std_errors)
    for_table.append(tuple(row))

  h1 = ','.join(['abs_error_th%03d' % x for x in range(101)])
  h2 = ','.join(['abs_error_th%03d_std' % x for x in range(101)])
  h3 = ','.join(['error_th%03d' % x for x in range(101)])
  h4 = ','.join(['error_th%03d_std' % x for x in range(101)])
  header_csv = 'alpha,time,time_std,%s,%s,%s,%s' % (h1, h2, h3, h4)
  mask_csv = ','.join(['%.2f'] * len(for_table[0]))

  eprint()
  with open(output_filename % ('', 'bft_raw_%s' % scorer_name), mode="w") as out:
    print(header_csv, file=out)
    for row in for_table:
      print(mask_csv % row, file=out)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('exp', type=str)
  parser.add_argument('scorer', type=str)
  parser.add_argument('--it', type=int, default=5)
  parser.add_argument('--msz', type=int, default=2000)
  parser.add_argument('--mlsz', type=int, default=500)
  args = parser.parse_args()
  RunExperiment(args.exp, args.scorer, args.it, args.msz, args.mlsz)
