from skimage import feature
import numpy as np
import mahotas
import cv2 as cv

fixed_size = tuple((500, 500))
bins = 8

# shape features


def fd_hu_moments(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    feature = cv.HuMoments(cv.moments(image)).flatten()
    return feature

# haralick texture


def fd_haralick(image):
    # convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# color histogram


def fd_histogram(image):
    # convert the image to HSV color-space
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv.calcHist([image], [0, 1, 2], None, [
        bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# local features


def fd_localBinaryPatters(image, numPoints=24, radius=8):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
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
    image = cv.imread(file)
    image = cv.resize(image, fixed_size)
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)
    fv_lbp = fd_localBinaryPatters(image)
    return np.hstack([fv_hu_moments, fv_haralick, fv_histogram, fv_lbp])


setting = "allFeatures"  # setting the setting heh
