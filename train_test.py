# -----------------------------------
# TRAINING OUR MODEL
# -----------------------------------
import h5py
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from helper import getGlobalFeatures
from helper import setting
import csv

num_trees = 100
test_size = 0.1
seed = 9
train_path = "dataset/train"
testPath = "dataset/mytest"
h5_data = 'output/data.h5'
h5_dataSplit = "output/dataSplit.h5"
h5_labels = 'output/labels.h5'
h5_labelsSplit = "output/labelsSplit.h5"
scoring = "accuracy"

h5_results = "output/results.h5"
h5_resultsSplit = "output/resultsSplit.h5"
csvOutputPath = 'output/csv/'

# get the training labels
train_labels = os.listdir(train_path)
train_labels.sort()

results = []
names = []

# import the feature vector, labels, open files for writing & reading
h5f_data = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')
h5f_dataSplit = h5py.File(h5_dataSplit, 'r')
h5f_labelSplit = h5py.File(h5_labelsSplit, 'r')

h5f_results = h5py.File(h5_results, 'a')
h5f_resultsSplit = h5py.File(h5_resultsSplit, 'a')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']
global_features_split_string = h5f_dataSplit['dataset_1']
global_labels_split_string = h5f_labelSplit['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)
global_features_split = np.array(global_features_split_string)
global_labels_split = np.array(global_labels_split_string)

h5f_data.close()
h5f_label.close()
h5f_labelSplit.close()
h5f_dataSplit.close()

# actual cross validation split testing
# split the training and testing data
model = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(
                                                                                              global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

# 10-fold cross validation
kfold = KFold(n_splits=10, random_state=seed)
cvScore = cross_val_score(
    model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
results.append(cvScore)
if h5f_results.get(setting) == None:
    h5f_results.create_dataset(setting, data=np.array(results))
h5f_results.close()
# print(results)
# names.append(name)
msg = '{} method mean result: {}'.format(setting, round(cvScore.mean(), 3))
print(msg)

# 80-20 split testing
# creating model for our manual testing
clf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
global_labels_split.sort()

# fitting data
clf.fit(global_features_split, global_labels_split)
testLabels = os.listdir(testPath)
testLabels.sort()
resultsSplit = []
resultsSplitPerClass = dict()

for i, testName in enumerate(testLabels):
    dir = os.path.join(testPath, testName)
    resultsSplitTmp = []  # tmp variable used to calculate success rate
    for file in os.listdir(dir):
        global_feature = getGlobalFeatures(dir+"/"+file)
        prediction = clf.predict([global_feature])[0]
        resultSplit = testName == testLabels[prediction]

        resultsSplitTmp.append(resultSplit)
    tmp = np.array(resultsSplitTmp).mean()
    resultsSplitPerClass[testName] = tmp
    resultsSplit.append(tmp)
    resultsSplitTmp.clear()

# print(np.array(resultsSplit).mean())
# print(resultsSplitPerClass)
# writing result to .csv file
with open(csvOutputPath+setting+'.csv', mode='w', newline='') as xdFile:
    xdWriter = csv.writer(xdFile, delimiter=';')
    xdWriter.writerow(resultsSplitPerClass.keys())
    xdWriter.writerow(resultsSplitPerClass.values())
print("END OF TRAIN_TEST")
