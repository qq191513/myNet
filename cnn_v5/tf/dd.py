import matplotlib
matplotlib.use('Qt5Agg') #这一句应该放在 import pyplot前面
import matplotlib.pyplot as plt

import numpy as np

plt.plot(np.arange(50))
plt.show()