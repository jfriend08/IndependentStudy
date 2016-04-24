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

def L_numericalGradient(m, x, y):
  limx, limy = m.shape
  if x > limx or y > limy:
    raise ValueError('Position outside matrix')
  # if x == y:
  #   raise ValueError('Cannot no self weighting')
  d = 1e-6
  f1, f2 = np.copy(m), np.copy(m)
  f1[x,y] += d
  f1[y,x] += d
  f2[x,y] -= d
  f2[y,x] -= d
  # l1, l2 = getLaplacianMatrix(f1), getLaplacianMatrix(f2)
  l1, l2 = scipy.sparse.csgraph.laplacian(f1, normed=True), scipy.sparse.csgraph.laplacian(f2, normed=True)
  # L1, d1= scipy.sparse.csgraph.laplacian(f1, normed=True, return_diag=True)
  # L2, d2= scipy.sparse.csgraph.laplacian(f2, normed=True, return_diag=True)
  return (l1-l2)/(2*d)

def L_analyticalGradient(m, x, y):
  limx, limy = m.shape
  if x > limx or y > limy:
    raise ValueError('Position outside matrix')
  # if x == y:
  #   raise ValueError('Cannot no self weighting')

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

def f1(x):
  return 4./(x)**2
def df1(x):
  return -8./(x**3)

def w_ij(i, j, sij, feature):
  diff = np.linalg.norm(feature[i]-feature[j])
  val = np.exp(-((diff/sij)**2))
  return val

def dw_ij(i, j, sij, feature):
  orig = w_ij(i, j, sij, feature)
  diff = np.linalg.norm(feature[i]-feature[j])
  return (2*orig*diff**2)/(sij**3)

def allDLoss(sigma, L, L_true, features):
  # m = np.array([[1,2,3,4,5],[2,1,2,3,4],[3,2,1,1,1],[4,3,1,1,1],[5,4,1,1,1]]).astype(float)
  accu = np.zeros(L.shape)
  dict_dw_ij, dict_dL = {}, {}
  for i in xrange(len(L)):
    for j in xrange(len(L)):
      if (i,j) in dict_dw_ij:
        dL = dict_dL[(i,j)]
        dw = dict_dw_ij[(i,j)]
      else:
        dL = L_analyticalGradient(L,i,j)
        dw = dw_ij(i,j,sigma[i,j],features)
        dict_dL[(i,j)], dict_dL[(j,i)] = dL, dL
        dict_dw_ij[(i,j)], dict_dw_ij[(j,i)] = dw, dw

      accu += dL * dw
  return -1 * (L_true - L) * accu

m = np.array([[1,2,3,4,5],[2,1,2,3,4],[3,2,1,1,1],[4,3,1,1,1],[5,4,1,1,1]]).astype(float)

# s = 2
# for i in xrange(len(m)):
#   for j in xrange(len(m)):
#     print dw_ij(i,j,s,m)-(w_ij(i,j,s+1e-6,m) - w_ij(i,j,s-1e-6,m))/(2*1e-6)<1e-10


# for i in xrange(len(m)):
#   for j in xrange(len(m)):
#     dL_num = L_numericalGradient(m, i,j)
#     dL_ana = L_analyticalGradient(m, i,j)
#     # print dL_num
#     # print dL_ana
#     # print '-----------------'
#     print "all diff are below 1e-10: ", np.all((dL_num-dL_ana)<1e-10)
#     print (dL_ana == dL_ana.T).all()
