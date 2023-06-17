import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

tot = np.load('out.npy')
print(tot.shape)

plt.figure(figsize=(5, 5))
sns.heatmap(tot[:, ::10], ax=plt.gca())
plt.yticks(np.arange(tot.shape[0])[::30], (np.arange(tot.shape[0]) / 29.98 * 5).round(1)[::30])
plt.savefig('out.png')
