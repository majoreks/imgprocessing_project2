import numpy as np
from matplotlib import pyplot
import h5py

h5_results = "output/results.h5"
h5f_results = h5py.File(h5_results, 'r')
#results = h5f_results['results1']
resultsMoments = np.array(h5f_results["moments"][:])
resultsHistogram = np.array(h5f_results["histogram"][:])
print(resultsMoments[0])
results = [resultsMoments[0], resultsHistogram[0]]
print(results)
# boxplot algorithm comparison
fig, ax = pyplot.subplots()
fig.suptitle('Machine Learning algorithm comparison')
#ax = fig.add_subplot(111)
#yplot.boxplot(results, vert=False)
ax.boxplot(results)
#ax.set_xticklabels(names)
#ax.set_yticklabels("all features")
pyplot.show()