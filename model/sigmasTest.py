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

sigmas = np.load('./sigmas/modelRun_Alpha3.npy')
sigmas = np.around(sigmas, decimals = 10)
for i in xrange(sigmas.shape[0]):
  for j in xrange(i+1, sigmas.shape[0]):
    if not sigmas[i,j] == sigmas[j,i]:
      print (i,j)
      print str(sigmas[i,j]), str(sigmas[j,i]), sigmas[i,j] == sigmas[j,i]



# sigmas0 = np.load('./sigmas/sigmas_0.npy')
# sigmas1 = np.load('./sigmas/sigmas_1.npy')
# sigmas2 = np.load('./sigmas/sigmas_2.npy')
# sigmas3 = np.load('./sigmas/sigmas_3.npy')

# mu, sigma, length = 10, 1, 22050/2 # mean, standard deviation, and sampling rate
# signal = []
# for i in xrange(20):
#   if i%2==0:
#     signal += list(np.random.normal(mu, sigma, length))
#   else:
#     signal += list(np.random.normal(-mu, sigma, length))
# signal = np.array(signal)
# cqt = np.transpose(librosa.cqt(signal))

# gm0 = RM.feature2GaussianMatrix(cqt, sigmas0)

# term = ['up', 'down']
# term2 = ['transitDown', 'transitUp']
# transitDelta = 50
# allInterval = []
# #Create interval base on the signal we created
# for i in xrange(20):
#   allInterval += [[np.array([length*i+transitDelta, length*(i+1)-transitDelta]).astype(float), term[i%2]]]
#   allInterval += [[np.array([length*(i+1)-transitDelta, length*(i+1)+transitDelta]).astype(float), term2[i%2]]]
# allInterval.pop()
# #Convert interval from time to frame
# def t2f(x):
#     return librosa.core.time_to_frames(x, sr=22050, hop_length=512, n_fft=None)
# allInterval = [[np.apply_along_axis(t2f, 0, elm[0]/22050), elm[1]] for elm in allInterval]
# m_true = RM.label2RecurrenceMatrix("../data/2.jams", gm0.shape[0], allInterval)


# L_true = scipy.sparse.csgraph.laplacian(m_true, normed=True)
# for i, sigma in enumerate([sigmas0, sigmas1, sigmas2, sigmas3]):
#   gm = RM.feature2GaussianMatrix(cqt, sigma)
#   L = scipy.sparse.csgraph.laplacian(gm, normed=True)
#   err = 0.5 * np.linalg.norm(L_true-L)**2
#   print "err"+str(i), str(err)



