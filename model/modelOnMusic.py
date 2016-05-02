import sys, scipy, os, warnings, librosa, argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.feature_extraction import image
from librosa.util import normalize
import numpy as np
import scipy.io.wavfile as wav


sys.path.append('../src')
import laplacian, gradient, plotGraph, librosaF, sigmaUpdate
import RecurrenceMatrix as RM

parser = argparse.ArgumentParser()
parser.add_argument("alpha", help="alpha for update step size")
parser.add_argument("sigmaMul", help="sigmaMul for base of sigmas")

args = parser.parse_args()
alpha = int(args.alpha)
sigmaMul = int(args.sigmaMul)

epco, res = 50, []
np.random.seed(123)

sigmaPath = "./sigmas/"
figurePath = "./fig/"
namePrefix = "test_newUpdate_Alpha" + str(alpha) + "_" + str(sigmaMul)
isBatch = False

sigmaPath += namePrefix + '/'
figurePath += namePrefix + '/'

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


#check sigma and figure path, creat if not exist
if not os.path.exists(sigmaPath):
  os.makedirs(sigmaPath)
if not os.path.exists(figurePath):
  os.makedirs(figurePath)


# sr, signal = mp32np('../data/audio/SALAMI_698.mp3')
# y = signal[:,0]
# print "Perform beat_track and cqt"
# tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
# cqt = librosa.cqt(y=y, sr=sr)
# print "saving cqt and beats... "
# np.save("cqt.npy", cqt)
# np.save("beats.npy", beats)

print "Loading cqt and beats... "
cqt = np.load('./cqt.npy')
beats = np.load('./beats.npy')
sr = 44100

print "Perform sync"
cqt_med, frameConversion = librosaF.sync(cqt, beats, aggregate=np.median)
cqt_med = cqt_med.T
# cqt_med = normalize(cqt_med, norm=2)
print "cqt_med.shape: %s" % str(cqt_med.shape)

print "Perform loadInterval2Frame"
interval = loadInterval2Frame("../data/anno/698/parsed/textfile1_uppercase.txt", sr, frameConversion)

base = 10
if sigmaMul >= base:
  raise ValueError('Sigma noise exceed base. Sigma has to be possitive')
sigmas = np.ones((cqt_med.shape[0], cqt_med.shape[0])) * base + np.random.rand(cqt_med.shape[0], cqt_med.shape[0])
sigmas = ((sigmas + sigmas.T)/2) * sigmaMul

print "sigmas"
print sigmas

gm = RM.feature2GaussianMatrix(cqt_med, sigmas) #(nSample, nFeature)
L = scipy.sparse.csgraph.laplacian(gm, normed=True)

print "gm.shape: ", gm.shape
print gm

m_true = RM.label2RecurrenceMatrix("../data/2.jams", gm.shape[0], interval)
L_true = scipy.sparse.csgraph.laplacian(m_true, normed=True)

print "L_true"
print L_true

np.save("./tempArray/cqt_med.npy", cqt_med)
np.save("./tempArray/gm.npy", gm)
np.save("./tempArray/L.npy", L)
np.save("./tempArray/m_true.npy", m_true)
np.save("./tempArray/L_true.npy", L_true)

print "gm.shape, m_true.shape: ", gm.shape, m_true.shape
print "gm is symmetric: ", (gm==np.transpose(gm)).all()
print "m_true is symmetric: ", (m_true==np.transpose(m_true)).all()

# plotGraph.plot2(namePrefix+"_R", m_true, "m_true", gm, "gm")
# plotGraph.plot2(namePrefix+"_L", L_true, "L_true", L, "L")

filename = figurePath + namePrefix + "_orig.png"
plotGraph.plot4(filename, m_true, "m_true", gm, "gm", L_true, "L_true", L, "L")

err = 0.5 * np.linalg.norm(L_true-L)**2
res += [err]
print "errors: ", str(err)

for ep in xrange(epco):
  '''update sigma'''
  if isBatch:
    sigmas = sigmaUpdate.batchUpdate(gm, L, L_true, cqt_med, sigmas, alpha, figurePath, namePrefix)
  else:
    sigmas = sigmaUpdate.updateEachOne(gm, L, L_true, cqt_med, sigmas, res, alpha, figurePath, namePrefix)
    # sigmas = updateTargetEachOne(gm, L, L_true, cqt_med, sigmas, res)

  print "sigmas:"
  print sigmas

  filename = sigmaPath + namePrefix + "_step" + str(ep) + ".npy"
  print "saving sigmas to: ", filename
  np.save(filename, sigmas)

  gm = RM.feature2GaussianMatrix(cqt_med, sigmas)
  L = scipy.sparse.csgraph.laplacian(gm, normed=True)
  print "gm"
  print gm

  filename = figurePath + "/" + namePrefix + "_epch" + str(ep)
  # plotGraph.plot2(filename+"R", m_true, "m_true", gm, "gm")
  # plotGraph.plot2(filename+"L", L_true, "L_true", L, "L")
  plotGraph.plot4(filename, m_true, "m_true", gm, "gm", L_true, "L_true", L, "L")

  err = 0.5 * np.linalg.norm(L_true-L)**2
  print "epoch: ", str(ep), " errors: ", str(err)
  res += [err]


  plotGraph.plotLine(figurePath + namePrefix + "_err", res, 'Error per epoch')

print res








# # print "Perform subsegment"
# # sub_beats = librosa.segment.subsegment(cqt, beats)
# # cqt_med_sub = librosa.util.sync(cqt, sub_beats, aggregate=np.median)


# # plt.figure()
# # plt.subplot(3, 1, 1)
# # librosa.display.specshow(librosa.logamplitude(cqt**2,ref_power=np.max),x_axis='time',sr=sr)
# # plt.colorbar(format='%+2.0f dB')
# # plt.title('CQT power, shape={}'.format(cqt.shape))

# # plt.subplot(3, 1, 2)
# # librosa.display.specshow(librosa.logamplitude(cqt_med**2,ref_power=np.max))
# # plt.colorbar(format='%+2.0f dB')
# # plt.title('Beat synchronous CQT power, ,shape={}'.format(cqt_med.shape))

# # plt.subplot(3, 1, 3)
# # librosa.display.specshow(librosa.logamplitude(cqt_med_sub**2,ref_power=np.max))

# # plt.colorbar(format='%+2.0f dB')
# # plt.title('Sub-beat synchronous CQT power, shape={}'.format(cqt_med_sub.shape))
# # plt.tight_layout()
# # plt.savefig('./test.png')