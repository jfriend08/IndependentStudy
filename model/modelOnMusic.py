import sys, scipy, os, librosa, warnings

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.feature_extraction import image
import numpy as np
import scipy.io.wavfile as wav

sys.path.append('../src')
import laplacian, gradient, plotGraph, librosaF
import RecurrenceMatrix as RM

epco, alpha, res, sigmaMul = 30, 2000, [], 10
np.random.seed(123)

sigmaPath = "./sigmas/"
figurePath = "./fig/"
namePrefix = "modelReal_Alpha" + str(alpha)

def mp32np(fname):
  oname = 'temp.wav'
  cmd = 'lame --decode {0} {1}'.format( fname,oname )
  os.system(cmd)
  return wav.read(oname)

def loadInterval2Frame(path, sr=22050, frameConversion=None):
  def t2f(x):
    return librosa.core.time_to_frames(x, sr=sr, hop_length=512, n_fft=None)

  def f2f(x):
    def getFrame(myin):
      if myin < 0 or myin > frameConversion[-1][1][1]:
        raise ValueError('frame index error')
      for newI, prevIs in frameConversion:
        if prevIs[0] <= myin <= prevIs[1]:
          return newI

    x = map(getFrame,x)

    if all([elm==x[0] for elm in x]):
      warnings.warn("Warning: Points are in the same frame after conversion")
      return []
    return x

  text_file = open(path, "r")
  tmpInt, interval = [], []

  #First pass. Get all annotation
  for line in text_file:
    tmpInt += [line.rstrip().split()]

  #Second pass. Get all intervals
  for i in xrange(len(tmpInt)-1):
    interval += [ [np.array(map(float,[tmpInt[i][0], tmpInt[i+1][0]])).astype(float), tmpInt[i][1]] ]

  #Time to frame conversion
  interval = [[np.apply_along_axis(t2f, 0, elm[0]), elm[1]] for elm in interval]

  if frameConversion != None:
    interval = [[np.apply_along_axis(f2f, 0, elm[0]), elm[1]] for elm in interval]
    print interval
    interval = [elm for elm in interval if elm[0].size!=0]
    print interval
  return interval



sr, signal = mp32np('../data/audio/SALAMI_698.mp3')
y = signal[:,0]
print "Perform beat_track and cqt"
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
cqt = librosa.cqt(y=y, sr=sr)

print "Perform sync"
cqt_med, frameConversion = librosaF.sync(cqt, beats, aggregate=np.median)
cqt_med = cqt_med.T

print "Perform loadInterval2Frame"
interval = loadInterval2Frame("../data/anno/698/parsed/textfile1_uppercase.txt", sr, frameConversion)

sigmas = np.random.rand(cqt_med.shape[0], cqt_med.shape[0])
sigmas = ((sigmas + sigmas.T)/2)*sigmaMul
gm = RM.feature2GaussianMatrix(cqt_med, sigmas) #(nSample, nFeature)
L = scipy.sparse.csgraph.laplacian(gm, normed=True)

print "gm.shape: ", gm.shape

m_true = RM.label2RecurrenceMatrix("../data/2.jams", gm.shape[0], interval)
L_true = scipy.sparse.csgraph.laplacian(m_true, normed=True)

print "gm.shape, m_true.shape: ", gm.shape, m_true.shape
print "gm is symmetric: ", (gm==np.transpose(gm)).all()
print "m_true is symmetric: ", (m_true==np.transpose(m_true)).all()

plotGraph.plot2('./realMusicRMatrix2', m_true, "m_true", gm, "gm")
plotGraph.plot2('./realMusicLaplacian2', L_true, "L_true", L, "L")

for i in xrange(epco):
  accu = gradient.allDLoss(sigmas, L, L_true, gm, cqt_med)
  sigmas = sigmas - alpha * accu
  sigmas = np.around(sigmas, decimals = 10)

  filename = sigmaPath + namePrefix + "_step" + str(i) + ".npy"
  print "saving sigmas to: ", filename
  np.save(filename, sigmas)

  gm = RM.feature2GaussianMatrix(cqt_med, sigmas)
  L = scipy.sparse.csgraph.laplacian(gm, normed=True)

  filename = figurePath + namePrefix + "_epch" + str(i) + ".png"
  plotGraph.plot2(filename, m_true, "m_true", gm, "gm")

  err = 0.5 * np.linalg.norm(L_true-L)**2
  res += [err]

print res








# print "Perform subsegment"
# sub_beats = librosa.segment.subsegment(cqt, beats)
# cqt_med_sub = librosa.util.sync(cqt, sub_beats, aggregate=np.median)


# plt.figure()
# plt.subplot(3, 1, 1)
# librosa.display.specshow(librosa.logamplitude(cqt**2,ref_power=np.max),x_axis='time',sr=sr)
# plt.colorbar(format='%+2.0f dB')
# plt.title('CQT power, shape={}'.format(cqt.shape))

# plt.subplot(3, 1, 2)
# librosa.display.specshow(librosa.logamplitude(cqt_med**2,ref_power=np.max))
# plt.colorbar(format='%+2.0f dB')
# plt.title('Beat synchronous CQT power, ,shape={}'.format(cqt_med.shape))

# plt.subplot(3, 1, 3)
# librosa.display.specshow(librosa.logamplitude(cqt_med_sub**2,ref_power=np.max))

# plt.colorbar(format='%+2.0f dB')
# plt.title('Sub-beat synchronous CQT power, shape={}'.format(cqt_med_sub.shape))
# plt.tight_layout()
# plt.savefig('./test.png')