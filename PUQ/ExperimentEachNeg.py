# See Experiment.py for more information.
# The difference in this file is that test sample has only one negative class.
# All negative classes are tested and a new column in the CSV file produced indicates the class.

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
from methodlist import methods
from tqdm import trange
from tqdm import tqdm

def eprint(*args, **kwargs):
  # print(*args, file=sys.stderr, **kwargs)
  pass

def RunExperiment(exp_name, method_name, niterations=5, max_sample_size=2000, max_labeled_size=500):
  exp = Exp[exp_name]
  dataset_filename = exp["input"]
  output_filename = exp["output"]
  class_feature = exp["class_feature"]
  positive_label = exp["positive_label"]
  method = methods[method_name]["func"]
  method_kargs = methods[method_name]["kargs"]
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
  _data = data_df.to_dict('registers')

  for_table = []
  for neg_label in negative_labels:
    data = [x for x in _data if x[class_feature] in {neg_label, positive_label}]
    for alpha in tqdm(list(np.linspace(0, 1, 11)), desc="alpha"):
      abs_errors = []
      errors = []
      ms_per_example = []
      for it in trange(niterations):
        shuffle(data)
        eprint('Iteration %d#' % (it + 1))
        for fold_i, (unlabeled, all_labeled) in zip(trange(5, desc="kfold"), KFold(5, data)):
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
          eprint('  #L %d #U %d' % (len(labeled), npos + nneg))
          eprint('       Actual c: %6.2f |    Actual alpha: %6.2f' % (actual_c, actual_alpha))
          tm_start = timer()
          pred_c, pred_alpha = method(sample, features, labeled_info, **method_kargs)
          tm_end = timer()
          eprint('    Predicted c: %6.2f | Predicted alpha: %6.2f' % (pred_c, pred_alpha))

          ms_per_example.append((tm_end - tm_start) * 1000 / len(sample))
          abs_errors.append(abs(actual_alpha - pred_alpha))
          errors.append(actual_alpha - pred_alpha)
      
      for_table.append((str(neg_label), 100 * alpha, 100 * np.mean(abs_errors), 100 * np.std(abs_errors), 100 * np.mean(errors), 100 * np.std(errors), np.mean(ms_per_example), np.std(ms_per_example)))

  header_csv = 'negative,alpha,mean_abs_error,mean_abs_error_std,mean_error,mean_error_std,time,time_std'
  mask_csv = ','.join(['%s'] + (['%.2f'] * 7))

  eprint()
  with open(output_filename % ('_EachNeg', method_name), mode="w") as out:
    print(header_csv, file=out)
    for row in for_table:
      print(mask_csv % row, file=out)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('exp', type=str)
  parser.add_argument('method', type=str)
  parser.add_argument('--it', type=int, default=5)
  parser.add_argument('--msz', type=int, default=2000)
  parser.add_argument('--mlsz', type=int, default=500)
  args = parser.parse_args()
  RunExperiment(args.exp, args.method, args.it, args.msz, args.mlsz)
