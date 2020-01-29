#-----------------------------------
# TRAINING OUR MODEL
#-----------------------------------
import h5py
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from helper import getGlobalFeatures
from helper import setting

warnings.filterwarnings('ignore')

#--------------------
# tunable-parameters
#--------------------
num_trees = 100
test_size = 0.1
seed      = 9
train_path = "dataset/train"
test_path  = "dataset/mytest"
h5_data    = 'output/data.h5'
h5_dataSplit = "output/dataSplit.h5"
h5_labels  = 'output/labels.h5'
h5_labelsSplit = "output/labelsSplit.h5"
scoring    = "accuracy"

h5_results = "output/results.h5"

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()

if not os.path.exists(test_path):
    os.makedirs(test_path)

# create all the machine learning models
model = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

# variables to hold the results and names
results = []
names   = []

# import the feature vector and trained labels
h5f_data  = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')
h5f_dataSplit = h5py.File(h5_dataSplit, 'r')
h5f_labelSplit = h5py.File(h5_labelsSplit, 'r')

h5f_results = h5py.File(h5_results, 'a')

global_features_string = h5f_data['dataset_1']
global_labels_string   = h5f_label['dataset_1']
global_features_split_string = h5f_dataSplit['dataset_1']
global_labels_split_string = h5f_labelSplit['dataset_1']

global_features = np.array(global_features_string)
global_labels   = np.array(global_labels_string)
global_features_split = np.array(global_features_split_string)
global_labels_split = np.array(global_labels_split_string)

h5f_data.close()
h5f_label.close()
h5f_labelSplit.close()
h5f_dataSplit.close()

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))

# 10-fold cross validation

kfold = KFold(n_splits=10, random_state=seed)
cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
results.append(cv_results)
if h5f_results.get(setting)==None:
    h5f_results.create_dataset(setting, data=np.array(results))
h5f_results.close()
print(results)
#names.append(name)
msg = "%s: %f (%f)" % ("xd", cv_results.mean(), cv_results.std())
print(msg)

import matplotlib.pyplot as plt

# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
global_labels_split.sort()
# fit the training data to the model
# print(global_labels_split)
# print(global_labels)
clf.fit(global_features_split, global_labels_split)
#print(train_labels)
# loop through the test images
test_labels = os.listdir(test_path)
test_labels.sort()
resultsSplit = []
resultsSplitPerClass = dict()

for i, test_name in enumerate(test_labels):
    dir = os.path.join(test_path, test_name)
    current_label = test_name
    resultsSplitTmp = []
    for file in os.listdir(dir):
        print(dir+"/"+file)
        global_feature = getGlobalFeatures(dir+"/"+file)
        prediction = clf.predict([global_feature])[0]
        #print(test_labels[prediction])
        resultSplit = current_label == test_labels[prediction] 
        resultsSplit.append(resultSplit)
        resultsSplitTmp.append(resultSplit)
        #print(resultSplit)
    resultsSplitPerClass[current_label] = np.array(resultsSplitTmp).mean()
    resultsSplitTmp.clear()
print(np.array(resultsSplit).mean())
print(resultsSplitPerClass)
# for file in glob.glob(test_path + "/*.jpg"):
#     # read the image
#     image = cv2.imread(file)
#     print(file)
#     # resize the image
#     #image = cv2.resize(image, fixed_size)

#     ####################################
#     # Global Feature extraction
#     ####################################
#     # fv_hu_moments = fd_hu_moments(image)
#     # fv_haralick   = fd_haralick(image)
#     # fv_histogram  = fd_histogram(image)
#     # fv_lbp        = fd_localBinaryPatters(image)

#     ###################################
#     # Concatenate global features
#     ###################################
#     global_feature = getGlobalFeatures(file)

#     # scale features in the range (0-1)
#     # scaler = MinMaxScaler(feature_range=(0, 1))
#     # rescaled_feature = scaler.fit_transform(global_feature.reshape(1,-1))

#     # predict label of test image
#     prediction = clf.predict([global_feature])[0]
#     print(global_labels_split[prediction])
#     # show predicted label on image
#     #cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

#     # display the output image
#     # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     # plt.show()