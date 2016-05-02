import sys, scipy, time
import numpy as np
from librosa.util import normalize

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
  nsample, nfeature = 100, 75
  sigma = np.random.rand(nsample, nsample) + 1
  sigma = (sigma + sigma.T)/2

  feature = np.random.rand(nsample, nfeature) + 2
  feature = normalize(feature, norm=2)
  for i in xrange(nsample):
    for j in xrange(nsample):
      dw_ana = gradient.dw_ij(i,j, sigma[i,j], feature)

      w1 = gradient.w_ij(i,j, sigma[i,j]+delta, feature)
      w2 = gradient.w_ij(i,j, sigma[i,j]-delta, feature)
      dw_num = (w1-w2)/(2*delta)

      print "is small: %s, dw_num: %s, dw_ana: %s" % (abs(dw_num-dw_ana) < 1e-10, dw_num, dw_ana)

def testDLOSSoverDsigma():
  nsample, nfeature = 322, 75
  delta = 1e-6

  # cqt_med = np.load('./tempArray/cqt_med.npy')
  cqt_med = np.random.rand(nsample, nfeature) + 200
  cqt_med = normalize(cqt_med, norm=2)

  sigmas = np.random.rand(nsample, nsample) + 500
  sigmas = ((sigmas + sigmas.T)/2)
  gm = RM.feature2GaussianMatrix(cqt_med, sigmas) #(nSample, nFeature)

  L = scipy.sparse.csgraph.laplacian(gm, normed=True)
  L_true = np.load("./tempArray/L_true.npy")

  for pos1 in xrange(nsample):
    for pos2 in xrange(nsample):
      if (L[pos1, pos2] - L_true[pos1, pos2]) < 1e-2:
        continue

      print "L_true[pos1,pos2]: %s and L[pos1,pos2]: %s at position" % (L_true[pos1,pos2], L[pos1,pos2])
      # print "start analytical ..."
      start_time = time.time()
      dJ12_anal = gradient.L_analyticalGradientII(gm, pos1,pos2, L_true, L, cqt_med, sigmas)
      # dJ12_anal = res.sum()
      print("--- Anal %s seconds ---" % (time.time() - start_time))
      # print "dJ12_anal: %s" % dJ12_anal

      # print "start numerical ..."
      start_time = time.time()
      sigmas1 = sigmas.copy()
      sigmas1[pos1,pos2] = sigmas1[pos1,pos2] + delta
      sigmas1[pos2,pos1] = sigmas1[pos2,pos1] + delta
      gm1 = RM.feature2GaussianMatrix(cqt_med, sigmas1) #(nSample, nFeature)
      L1 = scipy.sparse.csgraph.laplacian(gm1, normed=True)
      # print "L1[pos1,pos1]: %s, L1[pos2,pos2]: %s" % (L1[pos1,pos1], L1[pos2,pos2])
      J1 = 0.5 * (np.linalg.norm(L_true - L1))**2

      sigmas2 = sigmas.copy()
      sigmas2[pos1,pos2] = sigmas2[pos1,pos2] - delta
      sigmas2[pos2,pos1] = sigmas2[pos2,pos1] - delta
      gm2 = RM.feature2GaussianMatrix(cqt_med, sigmas2) #(nSample, nFeature)
      L2 = scipy.sparse.csgraph.laplacian(gm2, normed=True)
      # print "L2[pos1,pos1]: %s, L2[pos2,pos2]: %s" % (L2[pos1,pos1], L2[pos2,pos2])
      # print "L_true[pos1,pos1]: %s, L_true[pos2,pos2]: %s" % (L_true[pos1,pos1], L_true[pos2,pos2])
      J2 = 0.5 * (np.linalg.norm(L_true - L2))**2
      print("--- Num %s seconds ---" % (time.time() - start_time))
      dJ12_num = (J1-J2)/(2*delta)

      # print "start comparison ..."
      print "dJ12_anal: %s, dJ12_num: %s, diff percentage: %s" % (dJ12_anal, dJ12_num, (dJ12_anal-dJ12_num)/dJ12_anal)

testDLOSSoverDsigma()
