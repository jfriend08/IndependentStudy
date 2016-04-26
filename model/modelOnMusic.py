import sys, scipy, os, librosa

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


sr, signal = mp32np('../data/audio/SALAMI_698.mp3')
y = signal[:,0]
print "Perform beat_track and cqt"
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
cqt = librosa.cqt(y=y, sr=sr)

print "Perform sync"
cqt_avg, frameConversion = librosaF.sync(cqt, beats)
cqt_med, frameConversion = librosaF.sync(cqt, beats, aggregate=np.median)
print "frameConversion:"
print frameConversion

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