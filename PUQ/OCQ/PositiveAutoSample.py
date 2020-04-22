from sklearn.model_selection import KFold

def PAS(scorer, data, n_splits=10):
  kf = KFold(n_splits=n_splits, shuffle=True)
  scores = []
  for train_index, test_index in kf.split(data):
    train = data.iloc[train_index]
    test = data.iloc[test_index]
    scores += scorer(train, test)
  return scores