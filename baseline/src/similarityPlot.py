import pylab

def similarityPlot():
  R = np.corrcoef(img)
  pylab.pcolor(R)
  colorbar()
  show()