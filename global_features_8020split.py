# -----------------------------------
# GLOBAL FEATURE EXTRACTION
# -----------------------------------
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import h5py
from helper import getGlobalFeatures

# --------------------
# tunable-parameters
# --------------------
images_per_class = 80
imgNumbers2 = [66, 84, 75, 97, 82, 38]  # cross validation
imgNumbersTrain = [53, 67, 60, 78, 66, 30]
imNumbersTest = [13, 17, 15, 19, 16, 8]
fixed_size = tuple((500, 500))
train_path = "dataset/train"
train_path_picked = "dataset/mytrain"
h5_data = 'output/dataSplit.h5'
h5_labels = 'output/labelsSplit.h5'
bins = 8

# get the training labels
train_labels = os.listdir(train_path_picked)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels = []

# loop over the training data sub-folders
for i, training_name in enumerate(train_labels):
    # join the training data path and each species training folder
    dir = os.path.join(train_path_picked, training_name)

    # get the current training label
    current_label = training_name

    # loop over the images in each sub-folder
    for x in range(1, imgNumbersTrain[i]+1):
        # get the image file name

        if(x < 10):
            file = dir + "/l0" + str(x) + ".jpg"
        else:
            file = dir + "/l" + str(x) + ".jpg"
        # print(file)
        # read the image and resize it to a fixed-size
        # image = cv2.imread(file)
        # image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        # fv_hu_moments = fd_hu_moments(image)
        # fv_haralick = fd_haralick(image)
        # fv_histogram = fd_histogram(image)
        # fv_lbp = fd_localBinaryPatters(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = getGlobalFeatures(file)

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")

# get the overall feature vector size
print("[STATUS] feature vector size {}".format(
    np.array(global_features).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

# scale features in the range (0-1)
# scaler = MinMaxScaler(feature_range=(0, 1))
# rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(global_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")