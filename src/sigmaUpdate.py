import numpy as np
import sys, scipy, os, warnings, librosa, argparse
sys.path.append('../src')
import laplacian, gradient, plotGraph, librosaF
import RecurrenceMatrix as RM

def updateTargetEachOne(gm, L, L_true, cqt_med, sigmas, res, alpha, figurePath, namePrefix):
  '''
  ToDo: need to check its correctness
  '''
  updateC, loop = 0, 1000
  while loop > 0:
    loop -= 1

    print 'where the max diff in L ...'
    diffL = np.absolute(L_true-L)
    x,y = np.unravel_index(diffL.argmax(), diffL.shape)
    print "diffL: %s, L_true: %s, L: %s" % ((diffL[x,y]), L_true[x,y], L[x,y])
    print (x,y)

    dL = gradient.L_analyticalGradient(gm,x,y)
    dw = gradient.dw_ij(x,y,sigmas[x,y],cqt_med)
    print "dL(x,y): %s" %dL[x,y]
    print "dw: %s" % dw
    print 'cqt_med l2 diff is: %s' % np.linalg.norm(cqt_med[x]-cqt_med[y])
    print 'w[x,y] is: %s, sigmas[x,y] is: %s' % (gm[x,y], sigmas[x,y])

    diff = alpha * ( -1 * 2 * (L_true - L) * dL  * dw )

    print 'diff-alpha will be ...'
    maxdiff = np.absolute(diff)
    print "the alpha-diff at location is: %s" % maxdiff[x,y]
    x,y = np.unravel_index(maxdiff.argmax(), maxdiff.shape)
    print "max in diff is: %s, location is: %s" % (np.amax(maxdiff), (x,y))

    sigmas = sigmas - diff
    gm = RM.feature2GaussianMatrix(cqt_med, sigmas)
    L = scipy.sparse.csgraph.laplacian(gm, normed=True)

    print 'after update. L diff at location: %s is %s' % (str((x,y)), L_true[x,y] - L[x,y])
    print '\n'
  return sigmas

def updateEachOne(gm, L, L_true, cqt_med, sigmas, res, alpha, figurePath, namePrefix):
  updateC = 0
  # update each sigma and transfer to GM immediatly
  for i in xrange(gm.shape[0]):
    for j in xrange(i+1, gm.shape[0]):
      if abs(L_true[i,j] - L[i,j]) > 1e-2: #only update the sigma where L is large
        updateC += 1
        print "Update at: %s" % str((i,j))
        '''New update version'''
        # dJij_anal_matrix = gradient.L_analyticalGradientII(gm, i,j, L_true, L, cqt_med, sigmas)
        # dJij_anal = dJij_anal_matrix.sum()
        dJij_anal = gradient.L_numericalGradientII(gm, i,j, L_true, L, cqt_med, sigmas, cqt_med)
        sigmas[i,j] = sigmas[i,j] - alpha * dJij_anal
        sigmas[j,i] = sigmas[j,i] - alpha * dJij_anal

        '''Previous update version'''
        # dL = gradient.L_analyticalGradient(gm,i,j)
        # dw = gradient.dw_ij(i,j,sigmas[i,j],cqt_med)
        # diff = alpha * ( -1 * 2 * (L_true - L) * dL  * dw )

        # print 'alpha diff at', str((i,j))
        # maxdiff = np.absolute(diff)
        # x,y = np.unravel_index(maxdiff.argmax(), maxdiff.shape)
        # print np.amax(maxdiff)
        # print (x,y)
        # sigmas = sigmas - diff

        # gm = RM.feature2GaussianMatrix(cqt_med, sigmas)
        # L = scipy.sparse.csgraph.laplacian(gm, normed=True)

        # err = 0.5 * np.linalg.norm(L_true-L)**2
        # res += [err]
        # print "cur err: ", str(err)

        # if updateC % 1000 == 1:
        #   filename = figurePath + namePrefix + "_update" + str((i,j))
        #   plotGraph.plot1(filename+"Sigma", sigmas, "sigmas_"+str((i,j)))
        #   plotGraph.plot4(filename, m_true, "m_true", gm, "gm", L_true, "L_true", L, "L")
        #   plotGraph.plotLine(figurePath + namePrefix + "_err", res, 'Perplexity per Steps -- Validation')
          # plotGraph.plot2(filename+"R", m_true, "m_true", gm, "gm")
          # plotGraph.plot2(filename+"L", L_true, "L_true", L, "L")
  return sigmas

def batchUpdate(gm, L, L_true, cqt_med, sigmas, alpha, figurePath, namePrefix):
  accu = gradient.allDLoss(sigmas, L, L_true, gm, cqt_med)
  sigmas = sigmas - alpha * accu
  sigmas = (sigmas + sigmas.T)/2

  filename = figurePath + namePrefix + "_BatchUpdate"
  plotGraph.plot1(filename+"Sigma", sigmas, "sigmas")

  return sigmas