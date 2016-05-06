import numpy as np
import scipy.sparse, math, sys, math
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

def L_analyticalGradientII_getMatrix(m, x, y, L_true, L, features, sigma):
  '''
  @para {m}: Should have to be recurrenc matrix
  @return: analytical accumulation of all loss value changes w.r.t sigma change at [x,y]

  Same as L_analyticalGradientII, but return matrix
  '''
  limx, limy = m.shape

  if x > limx or y > limy:
    raise ValueError('Position outside matrix')

  g = m.copy()
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
        val =  m[i,j]/(2*(d[i]**1.5*d[j]**0.5))
        needUpdate = True
      elif j==x or j==y:
        val =  m[i,j]/(2*(d[j]**1.5*d[i]**0.5))
        needUpdate = True

      if needUpdate:
        val = (0 if np.isnan(val) else val)
        res[i,j] = dl * -1 * val * dw
        res[j,i] = dl * -1 * val * dw
        # accu = 2 * (dl * -1 * val * dw)
  return res

def L_analyticalGradientII(m, x, y, L_true, L, features, sigma):
  '''
  @para {m}: Should have to be recurrenc matrix
  @return: analytical accumulation of all loss value changes w.r.t sigma change at [x,y]

  This is function calculating:
  1. deriative of L w.r.t W at (x,y)
  2. times deriative of W at (x,y) w.r.t sigma at (x,y)
  3. times L_true, L difference ar (x,y)
  '''
  limx, limy = m.shape

  if x > limx or y > limy:
    raise ValueError('Position outside matrix')

  g = m.copy()
  np.fill_diagonal(g, 0)
  d = g.sum(axis = 1)

  accu = 0
  for i in xrange(limx):
    for j in xrange(i+1, limy):
      needUpdate = False
      dw = dw_ij(i,j,sigma[i,j],features)
      dl = L_true[i,j]-L[i,j]
      if (i==x and j==y) or (i==y and j==x):
        val = -1/((d[i]*d[j])**0.5) + m[i,j]*(d[i]+d[j])/(2*(d[i]*d[j])**1.5)
        needUpdate = True
      elif i==x or i==y:
        val =  m[i,j]/(2*(d[i]**1.5*d[j]**0.5))
        needUpdate = True
      elif j==x or j==y:
        val =  m[i,j]/(2*(d[j]**1.5*d[i]**0.5))
        needUpdate = True

      if needUpdate:
        val = (0 if np.isnan(val) else val)
        accu = 2 * (dl * -1 * val * dw)
  return accu

def L_numericalGradientII(m, pos1, pos2, L_true, L, features, sigmas, cqt_med):
  '''
  @para {m}: Should have to be recurrenc matrix
  @return: numerical accumulation of all loss value changes w.r.t sigma change at [x,y]

  This is the same objective as def L_analyticalGradientII, but using numerical approach
  '''

  delta = 1e-9
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
  '''
  @para {m}: Should have to be recurrenc matrix
  @return: numerical result of laplacian matrix after change of w[x,y]
  '''

  limx, limy = m.shape
  if x > limx or y > limy:
    raise ValueError('Position outside matrix')
  d = 1e-6
  f1, f2 = np.copy(m), np.copy(m)
  f1[x,y] += d
  f1[y,x] += d
  f2[x,y] -= d
  f2[y,x] -= d
  l1, l2 = scipy.sparse.csgraph.laplacian(f1, normed=True), scipy.sparse.csgraph.laplacian(f2, normed=True)
  return (l1-l2)/(2*d)

def L_analyticalGradient(m, x, y):
  '''
  @para {m}: Should have to be recurrenc matrix
  @return: analytical result of laplacian matrix after change of w[x,y]
  '''

  limx, limy = m.shape

  if x > limx or y > limy:
    raise ValueError('Position outside matrix')

  g = np.array(m)
  np.fill_diagonal(g, 0)
  d = g.sum(axis = 1)

  res = np.array(m)
  res.fill(0)
  for i in xrange(limx):
    for j in xrange(i+1, limy):
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

def w_ij(i, j, sij, feature):
  '''
  @feature: feture matrix with shape of (nSample, nFeature)
  @para {i,j}: time points from feature matrix

  @return: analytical result of w_ij
  '''

  diff = np.linalg.norm(feature[i]-feature[j])
  val = math.exp(-((diff/sij)**2))
  return val

def dw_ij(i, j, sij, feature):
  '''
  @feature: feture matrix with shape of (nSample, nFeature)
  @para {i,j}: time points from feature matrix

  @return: analytical result of derivitive of w_ij
  '''

  orig = w_ij(i, j, sij, feature)
  diff = np.linalg.norm(feature[i]-feature[j])
  return (2*orig*diff**2)/(sij**3)

def allDLoss(sigma, L, L_true, RMatrix, features):
  '''
  This may not be correct
  '''
  accu = np.zeros(L.shape)
  for i in xrange(len(L)):
    for j in xrange(i+1, len(L)): #no i==j, due to dw is all zero
      dL = L_analyticalGradient(RMatrix,i,j)
      dw = dw_ij(i,j,sigma[i,j],features)
      accu += 2 * (dL * dw) #due to its symmetric properity
  return -1 * (L_true - L) * accu
