from skimage import feature
import numpy as np
import mahotas
import cv2 as cv

# setting the feature at the bottom

size = tuple((500, 500))

# shape features
def fd_hu_moments(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    feat = cv.HuMoments(cv.moments(img)).flatten()
    return feat

# haralick texture
def fd_haralick(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# color histogram
def fd_histogram(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([img], [0, 1, 2], None, [
        8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv.normalize(hist, hist)
    return hist.flatten()

# local features
def fd_localBinaryPatters(image, numPoints=24, radius=8):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(
        gray, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# change this function according to which feature you want to check
def getGlobalFeatures(file):
    image = cv.imread(file)
    image = cv.resize(image, size)
    #fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    #fv_histogram = fd_histogram(image)
    #fv_lbp = fd_localBinaryPatters(image)
    return np.hstack([fv_haralick]) # don't forget to add feature vector (fv_) to the hstack here

setting = "haralickTexture"  # setting the setting heh
# possibilities:
# 'moments', 'histogram', 'haralickTexture', 'locBinPatterns', 'allFeatures'
