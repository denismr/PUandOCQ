def KFold(k, d):
  for i in range(k):
    tr = [x for (j, x) in enumerate(d) if j % k != i]
    te = [x for (j, x) in enumerate(d) if j % k == i]
    yield tr, te