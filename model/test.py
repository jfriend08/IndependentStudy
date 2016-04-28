import numpy as np
sigmas = np.load('./sigmas/modelReal_Alpha2000_step0.npy')
sigmas1 = np.load('./sigmas/modelReal_Alpha2000_step1.npy')

print "sigmas is symmetric: ",  (sigmas==sigmas.T).all()
print "sigmas1 is symmetric: ",  (sigmas1==sigmas1.T).all()

for i in xrange(sigmas.shape[0]):
  print (sigmas[i,:]==sigmas1[i,:]).all()
# for elm in sigmas:
#   print elm