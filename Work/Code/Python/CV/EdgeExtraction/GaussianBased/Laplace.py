import cv2

if '__main__' == __name__:

    ddepth = cv2.CV_16S
    kernel_size = 3

    image = cv2.imread ("asset/court_03.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    dst = cv2.Laplacian(gray, ddepth, ksize=kernel_size)

    # converting back to uint8
    result = cv2.convertScaleAbs(dst)

    cv2.imshow('LaplaceGaussEdge', result)
    cv2.waitKey(0)
    