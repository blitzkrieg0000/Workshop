import cv2 as cv2
import numpy as np

class Thresholder():
    def __init__(self) -> None:
        pass


    def PreProcess(func):
        def wrapper(self, *args, **kwargs):
            img = args[0].copy()
            return func(self, img)

        return wrapper
    

    @PreProcess
    def ThresholdBinary(self, image: np.ndarray, threshold=127, max_val=255, thres_type=cv2.THRESH_BINARY):
        ret, result = cv2.threshold(image, threshold, max_val, thres_type)
        return result


    @PreProcess
    def ThresholdAdaptiveMean(self, image: np.ndarray, max_val=127, block_size=11, c=2, thres_type=cv2.THRESH_BINARY):
        result = cv2.adaptiveThreshold(image, max_val, cv2.ADAPTIVE_THRESH_MEAN_C, thres_type, block_size, c)
        return result


    @PreProcess
    def ThresholdAdaptiveGauss(self, image: np.ndarray, max_val=127, block_size=11, c=2, thres_type=cv2.THRESH_BINARY):
        result = cv2.adaptiveThreshold(image, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thres_type, block_size, c)
        return result


    @PreProcess
    def ThresholdBinaryOtsu(self, image: np.ndarray, threshold=0, max_val=127):
        # #OTSU Threshold
        # hist = cv2.calcHist([image],[0],None,[256],[0,256])
        # hist_norm = hist.ravel()/hist.sum()
        # Q = hist_norm.cumsum()
        # bins = np.arange(256)
        # fn_min = np.inf
        # threshold = -1
        # for i in range(1,256):
        #     p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        #     q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        #     if q1 < 1.e-6 or q2 < 1.e-6:
        #         continue
        #     b1,b2 = np.hsplit(bins,[i]) # weights
        #     # finding means and variances
        #     m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        #     v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        #     # calculates the minimization function
        #     fn = v1*q1 + v2*q2
        #     if fn < fn_min:
        #         fn_min = fn
        #         threshold = i
        ret2, th2 = cv2.threshold(image, threshold, max_val, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return th2