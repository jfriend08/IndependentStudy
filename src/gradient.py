import numpy as np
import scipy.sparse, math, sys
from scipy.sparse import isspmatrix

sys.path.append('../src')
import RecurrenceMatrix as RM

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

def L_analyticalGradientII(m, x, y, L_true, L, features, sigma):
  '''
  @para {m}: Should have to be recurrenc matrix

  This is function calculating:
  1. deriative of L w.r.t W at (x,y)
  2. times deriative of W at (x,y) w.r.t sigma at (x,y)
  3. times L_true, L difference ar (x,y)
  '''
  limx, limy = m.shape

  if x > limx or y > limy:
    raise ValueError('Position outside matrix')
  # if x == y:
  #   raise ValueError('Cannot no self weighting')

  g = m.copy()
  # g = np.array(m)
  np.fill_diagonal(g, 0)
  d = g.sum(axis = 1)

  res = np.array(m)
  res.fill(0)
  for i in xrange(limx):
    for j in xrange(i+1, limy):
      needUpdate = False
      dw = dw_ij(i,j,sigma[i,j],features)
      dl = L_true[i,j]-L[i,j]
      if (i==x and j==y) or (i==y and j==x):
        val = -1/((d[i]*d[j])**0.5) + m[i,j]*(d[i]+d[j])/(2*(d[i]*d[j])**1.5)
        needUpdate = True
      elif i==x or i==y:
        val =  m[i,j]/(2*((d[i]**1.5) * (d[j]**0.5)))
        needUpdate = True
      elif j==x or j==y:
        val =  m[i,j]/(2*((d[j]**1.5) * (d[i]**0.5)))
        needUpdate = True

      if needUpdate:
        val = (0 if np.isnan(val) else val)
        # print "loc: %s, res[i,j]: %s, dw: %s, dl: %s, val: %s, m[i,j]: %s, d[i]: %s, d[j]: %s" % (str((i,j)), res[i,j], dw, dl, val, m[i,j], d[i], d[j])

        res[i,j] = dl * -1 * val * dw
        res[j,i] = dl * -1 * val * dw
  return res

def L_numericalGradientII(m, pos1, pos2, L_true, L, features, sigmas, cqt_med):
  '''
  @para {m}: Should have to be recurrenc matrix

  This is the same objective as def L_analyticalGradientII, but using numerical approach
  '''
  delta = 1e-8
  sigmas1 = sigmas.copy()
  sigmas1[pos1,pos2] = sigmas1[pos1,pos2] + delta
  sigmas1[pos2,pos1] = sigmas1[pos2,pos1] + delta
  gm1 = RM.feature2GaussianMatrix(cqt_med, sigmas1) #(nSample, nFeature)
  L1 = scipy.sparse.csgraph.laplacian(gm1, normed=True)
  J1 = 0.5 * (np.linalg.norm(L_true - L1))**2

  sigmas2 = sigmas.copy()
  sigmas2[pos1,pos2] = sigmas2[pos1,pos2] - delta
  sigmas2[pos2,pos1] = sigmas2[pos2,pos1] - delta
  gm2 = RM.feature2GaussianMatrix(cqt_med, sigmas2) #(nSample, nFeature)
  L2 = scipy.sparse.csgraph.laplacian(gm2, normed=True)
  J2 = 0.5 * (np.linalg.norm(L_true - L2))**2

  dJ_num = (J1-J2)/(2*delta)
  return dJ_num

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
  '''
  This is function calculating deriative of L r.w.t W at (x,y).
  '''
  limx, limy = m.shape

  if x > limx or y > limy:
    raise ValueError('Position outside matrix')
  # if x == y:
  #   raise ValueError('Cannot no self weighting')

  g = np.array(m)
  np.fill_diagonal(g, 0)
  d = g.sum(axis = 1)

  # mask = (d<10e-10)
  # d[mask] = 1
  # print mask

  res = np.array(m)
  res.fill(0)
  for i in xrange(limx):
    for j in xrange(i+1, limy):
      # print "d[i], d[j]", (d[i], d[j])
      if (i==x and j==y) or (i==y and j==x):
        val = -1/((d[i]*d[j])**0.5) + m[i,j]*(d[i]+d[j])/(2*(d[i]*d[j])**1.5)
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
  # print "diff", str(diff), " sij", sij, " val", val
  return val

def dw_ij(i, j, sij, feature):
  orig = w_ij(i, j, sij, feature)
  diff = np.linalg.norm(feature[i]-feature[j])
  return (2*orig*diff)/(sij**3)

def allDLoss(sigma, L, L_true, RMatrix, features):
  # m = np.array([[1,2,3,4,5],[2,1,2,3,4],[3,2,1,1,1],[4,3,1,1,1],[5,4,1,1,1]]).astype(float)
  accu = np.zeros(L.shape)
  for i in xrange(len(L)):
    for j in xrange(i+1, len(L)): #no i==j, due to dw is all zero
      dL = L_analyticalGradient(RMatrix,i,j)
      dw = dw_ij(i,j,sigma[i,j],features)
      accu += 2 * (dL * dw) #due to its symmetric properity
  return -1 * (L_true - L) * accu

# def allDLossII(sigma, L, L_true, RMatrix, features):
#   # m = np.array([[1,2,3,4,5],[2,1,2,3,4],[3,2,1,1,1],[4,3,1,1,1],[5,4,1,1,1]]).astype(float)
#   accu = np.zeros(L.shape)
#   [2*L_analyticalGradient(RMatrix,i,j)*dw_ij(i,j,sigma[i,j],features) for j in xrange(i+1,len(L)) for i in xrange(len(L))]
#   np.fromfunction(getVal, sigma.shape)
#   return accu

# m = np.array([[1,2,3,4,5],[2,1,2,3,4],[3,2,1,1,1],[4,3,1,1,1],[5,4,1,1,1]]).astype(float)

# m = np.random.rand(100, 100)*100
# m = (m + m.T)/2
# print "is symmetric: ", (m == m.T).all()
# s = 0.001
# print all([dw_ij(i,j,s,m)-(w_ij(i,j,s+1e-10,m) - w_ij(i,j,s-1e-6,m))/(2*1e-6)<1e-10 for j in xrange(len(m)) for i in xrange(len(m))])


# for i in xrange(len(m)):
#   for j in xrange(len(m)):
#     dL_num = L_numericalGradient(m, i,j)
#     dL_ana = L_analyticalGradient(m, i,j)
#     # print dL_num
#     # print dL_ana
#     # print '-----------------'
#     print "all diff are below 1e-10: ", np.all((dL_num-dL_ana)<1e-10)
#     print (dL_ana == dL_ana.T).all()
