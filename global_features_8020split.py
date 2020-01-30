from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import h5py
from helper import getGlobalFeatures

# paths
trainPath = "dataset/mytrain"
h5_data = 'output/dataSplit.h5'
h5_labels = 'output/labelsSplit.h5'

# get the training labels
trainLabels = os.listdir(trainPath)
trainLabels.sort()

globalFeatures = []
labels = []

print("Getting features for the 80-20 split set")
# actual feature extraction
for i, trainName in enumerate(trainLabels):
    # path
    dir = os.path.join(trainPath, trainName)

    # loop over the images in each sub-folder
    for file in os.listdir(dir):
        globalFeature = getGlobalFeatures(dir+'/'+file)

        # add results
        labels.append(trainName)
        globalFeatures.append(globalFeature)
    print("{} class processed".format(trainName))

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)

# save the feature vector using .h5 files
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(globalFeatures))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("Feature for 80-20 split extracted")
print("END OF global_features_8020")
