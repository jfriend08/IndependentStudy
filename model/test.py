import numpy as np
sigmas0 = np.load('./sigmas/modelReal_Alpha2000_step0.npy')
sigmas1 = np.load('./sigmas/modelReal_Alpha2000_step1.npy')
sigmas2 = np.load('./sigmas/modelReal_Alpha2000_step2.npy')

for i in xrange(sigmas0.shape[0]):
  print (sigmas0[i,:] == sigmas1[i,:]).all()
  print (sigmas1[i,:] == sigmas2[i,:]).all()
  print (sigmas0[i,:] == sigmas2[i,:]).all()
  print '----------------------------------'