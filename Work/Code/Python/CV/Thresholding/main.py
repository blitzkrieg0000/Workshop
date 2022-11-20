import cv2
from helper.EdgeDetector import EdgeDetector
from lib.Thresholder import Thresholder


if '__main__' == __name__:
    edgeDetector = EdgeDetector()
    thresholder = Thresholder()


    image = cv2.imread ("asset/court_2.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gürültü Azalt
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    result = edgeDetector.Prewitt(gray)
 
    # # Gürültü Azalt
    # result = cv2.medianBlur(result, 5)
    
    canvas = thresholder.ThresholdBinaryOtsu(result)

    cv2.imshow('', result)
    cv2.waitKey(0)
    
    cv2.imshow('', canvas)
    cv2.waitKey(0)