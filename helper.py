from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from skimage import feature
import numpy as np
import mahotas
import cv2
import os
import h5py
import cv2


def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
# feature-descriptor-2: Haralick Texture


def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram


def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [
                        bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


def fd_localBinaryPatters(image, numPoints=24, radius=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(
        gray, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    # return the histogram of Local Binary Patterns
    return hist

def getGlobalFeatures(file):
    image = cv2.imread(file)
    image = cv2.resize(image, fixed_size)
    #fv_hu_moments = fd_hu_moments(image)
    # fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)
    # fv_lbp = fd_localBinaryPatters(image)
    return np.hstack([fv_histogram])

fixed_size = tuple((500, 500))
bins = 8
setting = "histogram"