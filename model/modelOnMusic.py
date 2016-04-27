import sys, scipy, os, librosa, warnings

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.feature_extraction import image
import numpy as np
import scipy.io.wavfile as wav

sys.path.append('../src')
import laplacian, gradient, plotGraph, librosaF
import RecurrenceMatrix as RM

import os

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
    print interval
    interval = [[np.apply_along_axis(f2f, 0, elm[0]), elm[1]] for elm in interval]

  return interval



sr, signal = mp32np('../data/audio/SALAMI_698.mp3')
y = signal[:,0]
print "Perform beat_track and cqt"
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
cqt = librosa.cqt(y=y, sr=sr)

print "Perform sync"
cqt_med, frameConversion = librosaF.sync(cqt, beats, aggregate=np.median)

# print "Perform loadInterval2Frame"
# interval = loadInterval2Frame("../data/anno/698/parsed/textfile1_uppercase.txt", sr, frameConversion)
# print interval
# # print "frameConversion:"
# # print frameConversion


print "Perform subsegment"
sub_beats = librosa.segment.subsegment(cqt, beats)
cqt_med_sub = librosa.util.sync(cqt, sub_beats, aggregate=np.median)


plt.figure()
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.logamplitude(cqt**2,ref_power=np.max),x_axis='time',sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('CQT power, shape={}'.format(cqt.shape))

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.logamplitude(cqt_med**2,ref_power=np.max))
plt.colorbar(format='%+2.0f dB')
plt.title('Beat synchronous CQT power, ,shape={}'.format(cqt_med.shape))

plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.logamplitude(cqt_med_sub**2,ref_power=np.max))

plt.colorbar(format='%+2.0f dB')
plt.title('Sub-beat synchronous CQT power, shape={}'.format(cqt_med_sub.shape))
plt.tight_layout()
plt.savefig('./test.png')