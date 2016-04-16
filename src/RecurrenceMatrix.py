import numpy as np
import math

def construct(labels):
  a = np.ones((1, labels.shape[0]))[0]
  R = np.diag(a, 0)
  for i in xrange(len(labels)-1):
    label = labels[i]
    for j in [j for j in xrange(i+1, len(labels)) if labels[i] == labels[j]]:
        R[i][j] = 1
        R[j][i] = 1
  return R

def adjacentMatrix(size):
  res = np.array([ [0.0]*size for _ in xrange(size)])
  for i in xrange(size):
    if i-1 >=0:
      res[i][i-1] = 1
    if i+1 < size:
      res[i][i+1] = 1
  return res

def sumEachRow(m):
  r, c = m.shape
  res = np.zeros(r)
  for i in xrange(r):
    res[i] = m[i,:]
  return res

def getBalancedMiu(R, delta):
  d_R = np.apply_along_axis(sum, 1, R)
  d_delta = np.apply_along_axis(sum, 1, delta)
  return np.dot(d_R,d_R+d_delta)/np.power(np.linalg.norm(d_R + d_delta), 2)

def gaussianKernel(m, c=0.5):
  numSample, numFeature = m.shape
  res = np.array([ [0.0]*numSample for _ in xrange(numSample)])
  for i in xrange(numSample):
    for j in xrange(i, numSample):
      diff = np.power(np.linalg.norm(m[i,:] - m[j,:]), 2)
      val = np.exp((-1*diff/(2*c**2)))
      res[i][j] = val
      res[j][i] = val
  return res

def feature2GaussianMatrix(feature, sigma):
  nSample, nFeature = feature.shape
  a = np.ones((1, nSample))[0]
  m = np.diag(a, 0)
  for i in xrange(nSample):
    for j in xrange(i+1, nSample):
      diff = np.power(np.linalg.norm(feature[i]-feature[j]),2)
      val = np.exp(-diff/sigma**2)
      m[i,j] = val
      m[j,i] = val
  return m

# # m = np.zeros((40, 84))
# m = np.matrix([[1,2,3],[2,2,4], [1,2,3], [2,2,4], [1,2,3]])
# print feature2GaussianMatrix(m, 1)


