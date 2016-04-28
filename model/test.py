import numpy as np
sigmas = np.load('./sigmas/modelReal_Alpha2000_step0.npy')

print "sigmas is symmetric: ",  (sigmas==sigmas.T).all()
print sigmas