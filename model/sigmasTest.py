import numpy as np
import sys, scipy

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.feature_extraction import image
import librosa

sys.path.append('../src')
import laplacian
import RecurrenceMatrix as RM
import gradient

sigmas0 = np.load('./sigmas/sigmas_0.npy')
sigmas1 = np.load('./sigmas/sigmas_1.npy')

mu, sigma, length = 10, 1, 22050/2 # mean, standard deviation, and sampling rate
signal = []
for i in xrange(20):
  if i%2==0:
    signal += list(np.random.normal(mu, sigma, length))
  else:
    signal += list(np.random.normal(-mu, sigma, length))
signal = np.array(signal)
cqt = np.transpose(librosa.cqt(signal))

gm0 = RM.feature2GaussianMatrix(cqt, sigmas0)

term = ['up', 'down']
term2 = ['transitDown', 'transitUp']
transitDelta = 50
allInterval = []
#Create interval base on the signal we created
for i in xrange(20):
  allInterval += [[np.array([length*i+transitDelta, length*(i+1)-transitDelta]).astype(float), term[i%2]]]
  allInterval += [[np.array([length*(i+1)-transitDelta, length*(i+1)+transitDelta]).astype(float), term2[i%2]]]
allInterval.pop()
#Convert interval from time to frame
def t2f(x):
    return librosa.core.time_to_frames(x, sr=22050, hop_length=512, n_fft=None)
allInterval = [[np.apply_along_axis(t2f, 0, elm[0]/22050), elm[1]] for elm in allInterval]
m_true = RM.label2RecurrenceMatrix("../data/2.jams", gm0.shape[0], allInterval)


gm0 = RM.feature2GaussianMatrix(cqt, sigmas0)
gm1 = RM.feature2GaussianMatrix(cqt, sigmas1)
L0 = scipy.sparse.csgraph.laplacian(gm0, normed=True)
L1 = scipy.sparse.csgraph.laplacian(gm1, normed=True)
L_true = scipy.sparse.csgraph.laplacian(m_true, normed=True)

err0 = 0.5 * np.linalg.norm(L_true-L0)**2
err1 = 0.5 * np.linalg.norm(L_true-L1)**2

print "err0: ", str(err0)
print "err1: ", str(err1)


