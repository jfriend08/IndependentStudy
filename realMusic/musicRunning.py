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


'''import file'''
rate, signal = s.read('../data/TheBeatles-ComeTogether.wav')
print rate, signal.shape

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title('Signal Wave: track1')
plt.plot(signal[:,0])

plt.subplot(1, 2, 2)
plt.title('Signal Wave: track2')
plt.plot(signal[:,1])
plt.savefig('signal')

'''Construct R and delta from cqt'''
#kmeans for finding similar features among different time frame
from sklearn.cluster import MiniBatchKMeans
y = signal[:,0]
cqt = np.transpose(librosa.cqt(y))
print "cqt.shape:", cqt.shape
n, dim = cqt.shape
clf = MiniBatchKMeans(n_clusters=100, batch_size=int(n*0.1), max_iter=200, max_no_improvement=None).fit(cqt)
#construct recurrence matrix R
R = RM.construct(clf.labels_)
#construct delta matrix R
delta = RM.adjacentMatrix(R.shape[0])

print "R is symmetric: ", (R == np.transpose(R)).all(), R.shape #check is symmetric
print "delta is symmetric: ", (delta == np.transpose(delta)).all(), delta.shape #check is symmetric
print '-------------------------- --------------'

'''Constuct affinity matrix'''
miu = RM.getBalancedMiu(R, delta)
R_gau = RM.gaussianKernel(R)
delta_gau = RM.gaussianKernel(delta)
A = miu*np.dot(R, R_gau) + (1-miu)*np.dot(delta, delta_gau) #constructing affinity matrix
print "miu:", miu, "R_gau.shape: ", R_gau.shape, "A.shape: ", A.shape
print "A is symmetric: ", (A == np.transpose(A)).all()
print '----------------------------------------'

'''Save model'''
np.save("./model/R.npy", R)
np.save("./model/R_gau.npy", R_gau)
np.save("./model/delta.npy", delta)
np.save("./model/delta_gau.npy", delta_gau)
np.save("./model/A.npy", A)

'''Laplacian formula'''
Y3 = laplacian.getNormLaplacian(A, 25) #similarity matrix, top m eigenvectors
np.save("./model/Y3.npy", Y3)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.pcolor(R[:1000, :1000], cmap="viridis")
plt.colorbar()
plt.title('Matrix- R')
plt.xlabel('Time')
plt.ylabel('Time')

plt.subplot(1, 3, 2)
plt.pcolor(A[:1000, :1000], cmap="viridis")
plt.colorbar()
plt.title('Matrix- A')
plt.xlabel('Time')
plt.ylabel('Time')

plt.subplot(1, 3, 3)
plt.pcolor(Y3[:1000, :], cmap="viridis")
plt.colorbar()
plt.title('Eigenvectors')
plt.ylabel('Time')
plt.xlabel('eachY')
plt.savefig('allMatrix')


# YY = np.dot(Y3[:,:], np.transpose(Y3)[:,:])
# plt.figure(figsize=(15, 5))
# for m in xrange(10):
#     YY = np.dot(Y3[:,:m], Y3[:,:m].T)
#     plt.subplot(2, 5, m+1)
#     plt.pcolor(YY, cmap="viridis")
#     plt.colorbar()
#     plt.title('m = {0}'.format(m+1))
#     plt.xlabel('Time')
#     plt.ylabel('Time')
# plt.savefig('pairSimilarities')
# print '----------------------------------------'
