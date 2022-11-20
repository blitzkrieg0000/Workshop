import cv2

from lib.EdgeDetector import EdgeDetector

if '__main__' == __name__:

    edgeDetector = EdgeDetector()

    image = cv2.imread ("asset/court_2.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    r1 = edgeDetector.Sobel(gray)
    r2 = edgeDetector.Prewitt(gray)
    r3 = edgeDetector.Robert(gray)
    r4 = edgeDetector.Canny(gray)
    r5 = edgeDetector.Laplace(gray)


    canvas = cv2.bitwise_and(r2, r3)


    cv2.imshow('before', gray)    
    cv2.imshow('canvas', canvas)
    cv2.imshow('Sobel', cv2.resize(r1, (480, 640)))
    cv2.imshow('Prewitt', cv2.resize(r2, (480, 640)))
    cv2.imshow('Robert', cv2.resize(r3, (480, 640)))
    cv2.imshow('Canny', cv2.resize(r4, (480, 640)))
    cv2.imshow('Laplace', cv2.resize(r5, (480, 640)))

    cv2.waitKey(0)
