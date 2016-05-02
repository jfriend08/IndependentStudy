import sys, scipy
import numpy as np

sys.path.append('../src')
import laplacian, gradient, plotGraph, librosaF
import RecurrenceMatrix as RM

np.random.seed(123)
delta = 1e-6

def testDLoverDW():
  w = np.random.rand(15, 15) + 1
  w = (w + w.T)/2
  L = scipy.sparse.csgraph.laplacian(w, normed=True)

  for i in xrange(10):
    for j in xrange(10):
      dL_num = gradient.L_analyticalGradient(w, i,j )

      wc1, wc2 = w.copy(), w.copy()
      wc1[i,j] += delta
      wc1[j,i] += delta
      wc2[i,j] -= delta
      wc2[j,i] -= delta
      dL_ana1 = scipy.sparse.csgraph.laplacian(wc1, normed=True)
      dL_ana2 = scipy.sparse.csgraph.laplacian(wc2, normed=True)
      dL_ana = (dL_ana1 - dL_ana2)/(2*delta)

      print "All small: %s" % ((dL_ana - dL_num) < 1e-10).all()

def testDWoverDSigma():
  sigma = np.random.rand(15, 15) + 1
  sigma = (sigma + sigma.T)/2
  feature = np.random.rand(15, 15) + 1