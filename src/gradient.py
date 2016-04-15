import numpy as np
import scipy.sparse, math
from scipy.sparse import isspmatrix

def getLaplacianMatrix(m):
  g = np.array(m)
  np.fill_diagonal(g, 0)
  d = g.sum(axis = 1)

  def getDs(arr):
    return map(lambda x: 1./math.sqrt(d[x]), arr)

  def getVal(i,j):
    return -m*np.array(map(getDs, i))*np.array(map(getDs, j))

  res = np.fromfunction(getVal, m.shape)
  np.fill_diagonal(res, 1)
  return res

def numericalGradient(m, x, y):
  limx, limy = m.shape
  if x > limx or y > limy:
    raise ValueError('Position outside matrix')
  if x == y:
    raise ValueError('Cannot no self weighting')
  d = 1e-6
  f1, f2 = np.copy(m), np.copy(m)
  f1[x,y] += d
  f1[y,x] += d
  f2[x,y] -= d
  f2[y,x] -= d
  l1, l2 = getLaplacianMatrix(f1), getLaplacianMatrix(f2)
  # L1, d1= scipy.sparse.csgraph.laplacian(f1, normed=True, return_diag=True)
  # L2, d2= scipy.sparse.csgraph.laplacian(f2, normed=True, return_diag=True)
  return (l1-l2)/(2*d)

def analyticalGradient(m, x, y):
  limx, limy = m.shape
  if x > limx or y > limy:
    raise ValueError('Position outside matrix')
  if x == y:
    raise ValueError('Cannot no self weighting')

  g = np.array(m)
  np.fill_diagonal(g, 0)
  d = g.sum(axis = 1)

  res = np.array(m)
  res.fill(0)
  for i in xrange(limx):
    for j in xrange(i+1, limy):
      if (i==x and j==y) or (i==y and j==x):
        val = -1/(d[i]*d[j])**0.5 + m[i,j]*(d[i]+d[j])/(2*(d[i]*d[j])**1.5)
        res[i,j] = val
        res[j,i] = val
      elif i==x or i==y:
        val =  m[i,j]/(2*(d[i]**1.5*d[j]**0.5))
        res[i,j] = val
        res[j,i] = val
      elif j==x or j==y:
        val =  m[i,j]/(2*(d[j]**1.5*d[i]**0.5))
        res[i,j] = val
        res[j,i] = val
  return res


m = np.array([[1,2,3,4,5],[2,1,2,3,4],[3,2,1,1,1],[4,3,1,1,1],[5,4,1,1,1]]).astype(float)
# m = np.ones((5,5))
# print m
# print getLaplacianMatrix(m)

# L, d= scipy.sparse.csgraph.laplacian(m, normed=True, return_diag=True)
# print L
dL_num = numericalGradient(m, 1,2)
dL_ana = analyticalGradient(m, 1,2)
print "all diff are below 1e-10: ", np.all((dL_num-dL_ana)<1e-10)
