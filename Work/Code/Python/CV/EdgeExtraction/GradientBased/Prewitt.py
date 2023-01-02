import cv2
import numpy as np

if '__main__' == __name__:
    image = cv2.imread ("asset/court_03.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    #! Prewitt Kenar Çıkartma
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(gray, -1, kernelx)
    img_prewitty = cv2.filter2D(gray, -1, kernely)


    # İki görüntüyü blendle
    grad = cv2.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)

    cv2.imshow('Prewitt_X', img_prewittx)
    cv2.waitKey(0)

    cv2.imshow('Prewitt_Y', img_prewitty)
    cv2.waitKey(0)

    cv2.imshow('Prewitt_X_Y_0.5', grad)
    cv2.waitKey(0)
    