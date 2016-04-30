import subprocess
import numpy as np


job = 's48'
time = "15:00:00"
ram = "8GB"
l_cmd = "walltime=%s,mem=%s"%(time, ram)
for alpha in xrange(100, 500, 100):
  for sigmaMul in xrange(500, 1000, 100):
    jobName = 'modelOnMusic_' + str(alpha) + "_" + str(sigmaMul)
    subprocess.call('qsub runModel.pbs -N {0} -q {1} -l {2} -v alpha={3},sigmaMul={4}'.format(
            jobName, job, l_cmd, alpha, sigmaMul), shell=True)