from random import random
from random import randrange
from math import floor
import numpy as np

def RndIntQnties(bins, sample_size):
  weights = [random() for x in range(bins)]
  sum_weights = sum(weights)
  quantities = [floor(sample_size * x / sum_weights) for x in weights]
  total = sum(quantities)
  # fix flooring artifacts
  for _ in range(total, sample_size):
    quantities[randrange(0, bins)] += 1
  return quantities

def RndPropWithAfterLimit(individual_max_sizes, max_size):
  individual_max_sizes = np.array(individual_max_sizes)
  weights = np.random.uniform(size = len(individual_max_sizes))
  weights /= np.sum(weights)
  left = min(np.min(individual_max_sizes), max_size)
  right = max_size
  while right >= left:
    middle = (right + left) // 2
    sizes = np.floor(weights * middle)
    check = np.prod(sizes <= individual_max_sizes) == 1
    if check:
      found = middle
      left = middle + 1
    else:
      right = middle - 1
  return [int(x) for x in np.floor(weights * found)]


if __name__ == "__main__":
  a = np.floor(np.random.uniform(low=10, high=20, size=10))
  print(a)
  res = RndPropWithAfterLimit(a, 200)
  print(res)
  print(sum(res))