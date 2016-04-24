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

mu, sigma, length = 10, 1, 22050 # mean and standard deviation

signal = []
for i in xrange(20):
    if i%2==0:
        signal += list(np.random.normal(mu, sigma, length))
    else:
        signal += list(np.random.normal(-mu, sigma, length))


signal = np.array(signal)
plt.figure(figsize=(15, 5))
plt.title('Signal Wave...')
plt.plot(signal)
plt.ylim((-15,15))
plt.savefig('./fig/signal.png')

############################################################
cqt = np.transpose(librosa.cqt(signal))
print signal.shape, cqt.shape

sigmas = np.random.rand(cqt.shape[0], cqt.shape[0])
sigmas = (sigmas + sigmas.T)/2
gm = RM.feature2GaussianMatrix(cqt, sigmas)
print gm.shape
print "is symmetric: ", (gm==np.transpose(gm)).all()

############################################################
term = ['up', 'down']
term2 = ['transitDown', 'transitUp']
transitDelta = 50
allInterval = []

#Create interval base on the signal we created
for i in xrange(20):
    allInterval += [[np.array([22050*i+transitDelta, 22050*(i+1)-transitDelta]).astype(float), term[i%2]]]
    allInterval += [[np.array([22050*(i+1)-transitDelta, 22050*(i+1)+transitDelta]).astype(float), term2[i%2]]]
allInterval.pop()

#Convert interval from time to frame
def t2f(x):
    return librosa.core.time_to_frames(x, sr=22050, hop_length=512, n_fft=None)
allInterval = [[np.apply_along_axis(t2f, 0, elm[0]/22050), elm[1]] for elm in allInterval]
print allInterval

m_true = RM.label2RecurrenceMatrix("../data/2.jams", gm.shape[0], allInterval)

############################################################
L = scipy.sparse.csgraph.laplacian(gm, normed=True)
L_true = scipy.sparse.csgraph.laplacian(m_true, normed=True)
err = 0.5 * np.linalg.norm(L_true-L)**2
epco, alpha, res = 5, 0.1, []

res += [err]
for i in xrange(epco):
  accu = gradient.allDLoss(sigmas, L, L_true, cqt)
  sigmas = sigmas - alpha * accu

  filename = "./sigmas/sigmas_" + i + ".npy"
  np.save(filename, sigmas)

  gm = RM.feature2GaussianMatrix(cqt, sigmas)
  L = scipy.sparse.csgraph.laplacian(gm, normed=True)
  err = 0.5 * np.linalg.norm(L_true-L)**2
  res += [err]
print res




