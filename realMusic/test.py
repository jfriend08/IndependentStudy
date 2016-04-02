import numpy as np
import sys, scipy, os

import librosa
from scipy.io import wavfile

from pylab import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')

sys.path.append('../src')
import laplacian
import RecurrenceMatrix as RM
import similarityMatrix as SM

o3 = SM.getOffDiagMatrixIII(3,5)

from sklearn.cluster import MiniBatchKMeans
# cqt = np.transpose(librosa.cqt(o3))
# print "cqt.shape:", cqt.shape
n, dim = o3.shape
clf = MiniBatchKMeans(n_clusters=100, batch_size=int(n*0.1), max_iter=200, max_no_improvement=None).fit(o3)
#construct recurrence matrix R
R = RM.construct(clf.labels_)
#construct delta matrix R
delta = RM.adjacentMatrix(R.shape[0])

print "R.shape: ", R.shape, "delta.shape: ", delta.shape



'''Constuct affinity matrix'''
miu = RM.getBalancedMiu(R, delta)
R_gau = RM.gaussianKernel(R)
delta_gau = RM.gaussianKernel(delta)
part1, part2 = np.dot(R, R_gau), np.dot(delta, delta_gau)
# part1, part2 = miu*np.dot(R, R_gau), (1-miu)*np.dot(delta, delta_gau)
for i in xrange(len(part2)):
  print list(part2[i,:10])

A = part1 + part2 #constructing affinity matrix
print "miu:", miu, "R_gau.shape: ", R_gau.shape, "A.shape: ", A.shape

print "delta_gau is symmetric: ", (delta_gau == np.transpose(delta_gau)).all()
print "R_gau is symmetric: ", (R_gau == np.transpose(R_gau)).all()
print "part1 is symmetric: ", (part1 == np.transpose(part1)).all()
print "part2 is symmetric: ", (part2 == np.transpose(part2)).all()
print "A is symmetric: ", (A == np.transpose(A)).all()
print '----------------------------------------'

np.save("./model/A", A)
A2 = np.load("./model/A.npy")
print A2