import cv2
import numpy as np


if '__main__' == __name__:
    image = cv2.imread ("asset/court_03.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gürültüyü azaltmak için
    gray = cv2.GaussianBlur(gray, (13, 13), 0)

    #! Sobel Kenar Çıkartma
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    # Dikey Çizgiler
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    # Yatay Çizgiler
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    print(grad_y, "\n")

    # Normalize Et -> 0 ile 255 arasında değerlere sığdır.
    abs_grad_x = cv2.convertScaleAbs(grad_x)


    abs_grad_y = cv2.convertScaleAbs(grad_y)

    print(abs_grad_y, "\n")

    # İki görüntüyü blendle
    grad = cv2.addWeighted(abs_grad_x, 0.7, abs_grad_y, 0.7, 0)

    print(grad, "\n")

    cv2.imshow('SOBEL_X', abs_grad_x)
    cv2.waitKey(0)

    cv2.imshow('SOBEL_Y', abs_grad_y)
    cv2.waitKey(0)

    cv2.imshow('SOBEL_X_Y_0.5', grad)
    cv2.waitKey(0)
    