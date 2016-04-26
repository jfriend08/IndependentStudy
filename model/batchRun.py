import subprocess
import numpy as np


job = 's48'
time = "15:00:00"
ram = "8GB"
l_cmd = "walltime=%s,mem=%s"%(time, ram)
for alpha in xrange(5,30,5):
  jobName = 'model_alpha' + str(alpha)
  subprocess.call('qsub runModel.pbs -N {0} -q {1} -l {2} -v alpha={3}'.format(
          jobName, job, l_cmd, d, jobName+"_", layer, maxNorm), shell=True)