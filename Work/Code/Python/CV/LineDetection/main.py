import cv2
from helper.EdgeDetector import EdgeDetector
from helper.Thresholder import Thresholder
from lib.LineDetector import LineDetector


if '__main__' == __name__:
    edgeDetector = EdgeDetector()
    thresholder = Thresholder()
    lineDetector = LineDetector()


    image = cv2.imread ("asset/court_1.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gürültü Azalt
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresholded = edgeDetector.Prewitt(gray)
 
    # # Gürültü Azalt
    # result = cv2.medianBlur(result, 5)
    
    canvas = thresholder.ThresholdBinaryOtsu(thresholded)
    lines = lineDetector.HoughLineTransformProbabilistic(canvas)
    result = lineDetector.DrawHoughLines(image, lines, True)

    cv2.imshow('', gray)
    cv2.waitKey(0)

    cv2.imshow('', result)
    cv2.waitKey(0)
    
