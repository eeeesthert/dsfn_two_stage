import numpy as np
from HM_ransac import HM_ransac
from utils import plot_matches, cv_draw_matches
import cv2 as cv

def comp_KR(im1, im2, X1, X2, display=True):
    H, ok, score = HM_ransac(X1, X2, 500, 0.1)
    print("Number of inliers: ", score)
    
    if score < 4 or not np.isfinite(H).all():
        raise ValueError("Insufficient RANSAC inliers")
    if display:
        img3 = cv_draw_matches(im1, im2, X1, X2, ok)
        cv.imshow("Ransac's results", img3)
        cv.waitKey(1)
    # plt.figure()
    # plot_matches(im1, im2, X1, X2, ok)
    # plt.title("Ransac's results")
    # plt.show()

    X1_ok = X1[:, ok]
    X2_ok = X2[:, ok]

    return H, X1_ok, X2_ok



