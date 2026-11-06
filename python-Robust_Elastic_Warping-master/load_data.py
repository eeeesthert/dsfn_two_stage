import cv2 as cv
import numpy as np

def load_data(im1, im2):
    
    sift = cv.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    gray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    kp1, ds1 = sift.detectAndCompute(gray1, None)
    gray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    kp2, ds2 = sift.detectAndCompute(gray2, None)

    if ds1 is None or ds2 is None or len(ds1) < 2 or len(ds2) < 2:
        raise ValueError("Insufficient SIFT descriptors")
    matches = flann.knnMatch(ds1, ds2, k=2)
    # store all the good matches as per Lowe's ratio test
    good = []
    for neighbors in matches:
        if len(neighbors) < 2:
            continue
        m, n = neighbors
        if m.distance < 0.7 * n.distance:
            good.append([m.queryIdx, m.trainIdx])
    
    if len(good) < 4:
        raise ValueError("Fewer than four SIFT matches")
    srcpoints = np.float32([kp1[m[0]].pt for m in good])
    dstpoints = np.float32([kp2[m[1]].pt for m in good])
    fp = np.vstack((srcpoints.T, np.ones((1, len(good)))))
    tp = np.vstack((dstpoints.T, np.ones((1, len(good)))))

    return fp, tp
