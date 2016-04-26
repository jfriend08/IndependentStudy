import numpy as np
import matplotlib.pyplot as plt

def plot2(path, obj1, name1, obj2, name2):
  plt.figure(figsize=(15, 5))
  plt.subplot(1, 2, 1)
  plt.pcolor(obj1, cmap="viridis")
  plt.colorbar()
  plt.title(name1)

  plt.subplot(1, 2, 2)
  plt.pcolor(obj2, cmap="viridis")
  plt.colorbar()
  plt.title(name2)
  plt.savefig(path)