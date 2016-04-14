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


m = np.array([[1,2,3,4,5],[2,1,2,3,4],[3,2,1,1,1],[4,3,1,1,1],[5,4,1,1,1]])
# m = np.ones((5,5))
print m
print getLaplacianMatrix(m)

L, d= scipy.sparse.csgraph.laplacian(m, normed=True, return_diag=True)
print L