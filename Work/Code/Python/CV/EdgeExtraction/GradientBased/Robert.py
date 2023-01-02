import cv2
import numpy as np


if '__main__' == __name__:
    image = cv2.imread ("asset/court_03.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('float64')
    gray/=255.0

    roberts_cross_r = np.array( 
                                [
                                    [1, 0 ],
                                    [0, -1 ]
                                ] 
                              )
    
    roberts_cross_l = np.array( 
                                [
                                    [ 0, 1 ],
                                    [-1, 0 ]
                                ] 
                              )


    line_r = cv2.filter2D(gray, -1, roberts_cross_r)
    line_l = cv2.filter2D(gray, -1, roberts_cross_l)


    # İki görüntüyü blendle
    grad = np.sqrt(np.square(line_r) + np.square(line_l))
    grad = grad*255
    grad = cv2.convertScaleAbs(grad)


    cv2.imshow('Robert_R', line_r)
    cv2.waitKey(0)

    cv2.imshow('Robert_L', line_l)
    cv2.waitKey(0)

    cv2.imshow('Robert', grad)
    cv2.waitKey(0)
    

