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

mu, sigma, length = 10, 1, 22050/2 # mean, standard deviation, and sampling rate

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
  allInterval += [[np.array([length*i+transitDelta, length*(i+1)-transitDelta]).astype(float), term[i%2]]]
  allInterval += [[np.array([length*(i+1)-transitDelta, length*(i+1)+transitDelta]).astype(float), term2[i%2]]]
allInterval.pop()

#Convert interval from time to frame
def t2f(x):
    return librosa.core.time_to_frames(x, sr=22050, hop_length=512, n_fft=None)
allInterval = [[np.apply_along_axis(t2f, 0, elm[0]/22050), elm[1]] for elm in allInterval]
print allInterval

m_true = RM.label2RecurrenceMatrix("../data/2.jams", gm.shape[0], allInterval)
print "m_true is symmetruc: ", (m_true==m_true.T).all()
############################################################
L = scipy.sparse.csgraph.laplacian(gm, normed=True)
L_true = scipy.sparse.csgraph.laplacian(m_true, normed=True)

print "L diag all 1: ", all([L[i,i] == 1 for i in xrange(L.shape[0])])
print "L_true diag all 1: ", all([L_true[i,i] == 1 for i in xrange(L_true.shape[0])])

err = 0.5 * np.linalg.norm(L_true-L)**2
epco, alpha, res = 5, 0.1, []
print "hi"
res += [err]

filename = "./fig/L_orignalAll.png"
plt.figure(figsize=(15, 5))
plt.subplot(2, 2, 1)
plt.pcolor(L_true, cmap="viridis")
plt.colorbar()
plt.title('L_true')

plt.subplot(2, 2, 2)
plt.pcolor(L, cmap="viridis")
plt.colorbar()
plt.title('L')

plt.subplot(2, 2, 3)
plt.pcolor(m_true, cmap="viridis")
plt.colorbar()
plt.title('m_true')

plt.subplot(2, 2, 4)
plt.pcolor(gm, cmap="viridis")
plt.colorbar()
plt.title('m')
plt.savefig(filename)

for i in xrange(epco):
  accu = gradient.allDLoss(sigmas, L, L_true, cqt)
  sigmas = sigmas - alpha * accu

  filename = "./sigmas/sigmas_" + str(i) + ".npy"
  print "saving sigmas to: ", filename
  np.save(filename, sigmas)

  gm = RM.feature2GaussianMatrix(cqt, sigmas)
  L = scipy.sparse.csgraph.laplacian(gm, normed=True)

  filename = "./fig/allL_" + str(i) + ".png"
  plt.figure(figsize=(15, 5))
  plt.subplot(2, 2, 1)
  plt.pcolor(L_true, cmap="viridis")
  plt.colorbar()
  plt.title('L_true')

  plt.subplot(2, 2, 2)
  plt.pcolor(L, cmap="viridis")
  plt.colorbar()
  plt.title('L')

  plt.subplot(2, 2, 3)
  plt.pcolor(m_true, cmap="viridis")
  plt.colorbar()
  plt.title('m_true')

  plt.subplot(2, 2, 4)
  plt.pcolor(gm, cmap="viridis")
  plt.colorbar()
  plt.title('m')
  plt.savefig(filename)

  err = 0.5 * np.linalg.norm(L_true-L)**2
  res += [err]
print res




