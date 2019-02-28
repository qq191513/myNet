import numpy as np
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

y = np.array([1,1,2,2])
pred = np.array([0.1,0.4,0.35,0.8])
fpr, tpr, thresholds = roc_curve(y, pred, pos_label=2)
print(fpr)
print(tpr)
print(thresholds)

from sklearn.metrics import auc
print(auc(fpr, tpr))
plt.plot(fpr, tpr)
plt.show()

