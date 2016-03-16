import numpy as np

def construct(labels):
  a = np.ones((1, labels.shape[0]))[0]
  R = np.diag(a, 0)
  for i in xrange(len(labels)-1):
    label = labels[i]
    for j in [j for j in xrange(i+1, len(labels)) if labels[i] == labels[j]]:
        R[i][j] = 1
        R[j][i] = 1
  return R
