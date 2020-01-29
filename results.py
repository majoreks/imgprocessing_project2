import numpy as np
from matplotlib import pyplot
import h5py
import csv

h5_results = 'output/results.h5'
csvPath = 'output/csv/'
resultsOutputPath = 'output/resultsSplit.txt'

names = ['moments', 'histogram', 'haralickTexture',
         'locBinPatterns', 'allFeatures']

f = open(resultsOutputPath, 'w')
h5f_results = h5py.File(h5_results, 'r')
results = []
resultsSplit = dict()
leaves = None
for j, name in enumerate(names):
    # reading data from .h5 file
    results.append(np.array(h5f_results[name][:])[0])
    # reading data from csv files
    with open(csvPath+name+'.csv', mode='r') as xdFile:
        xdReader = csv.reader(xdFile, delimiter=';')
        for i, row in enumerate(xdReader): 
            if(j == 0 and i == 0):
                leaves = list(row) # not very smart but works in our case
            if(i == 0):
                continue
            resultsSplit[name] = np.array(list(row))

# printing closs validation results
msg = 'One big data set with cross validation results:\n'
print(msg)
f.write(msg+'\n')
for i, name in enumerate(names):
    msg = '{}: {}'.format(name, round(results[i].mean(), 3))
    print(msg)
    f.write(msg+'\n')

# printing results for 80-20 split 
tmp2 = 0 # tmp variable used to calculate result for the whole set
msg = '\n80-20 split results:\nLeaves: {}\n'.format(leaves)
print(msg)
f.write(msg+'\n')
for name in resultsSplit.keys():
    msg = '{} method:'.format(name)
    print(msg)
    f.write(msg+'\n')
    tmp1 = 0 # tmp variable used to calculate result for given method
    for j, leaf in enumerate(leaves):
        tmp = float(resultsSplit.get(name)[j]) # result for a given leaf
        msg = '{}: {}'.format(leaf, round(tmp, 3))
        print(msg)
        f.write(msg+'\n')
        tmp1 = tmp1 + tmp
    tmp1 = tmp1/len(leaves)
    msg = 'mean result for this method: {}\n'.format(round(tmp1, 3))
    print(msg)
    f.write(msg+'\n')
    tmp2 = tmp2 + tmp1

tmp2 = tmp2/len(resultsSplit)
msg = 'mean result for the whole dataset (for 80-20 split): {}'.format(
    round(tmp2, 3))
print(msg)
f.write(msg+'\n')

# plotting results for different features
fig, ax = pyplot.subplots()
fig.suptitle('Machine Learning algorithm comparison')
ax.boxplot(results, vert=False)
ax.set_yticklabels(names)
pyplot.show()
f.close()
