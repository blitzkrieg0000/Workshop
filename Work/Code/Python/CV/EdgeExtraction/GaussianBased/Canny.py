import cv2

if '__main__' == __name__:


    image = cv2.imread ("asset/court_1.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gürültüyü azaltmak için
    gray = cv2.GaussianBlur(gray, (5, 5), 0)


    result = cv2.Canny(gray, 100, 200)


    cv2.imshow('CannyEdge', gray)
    cv2.waitKey(0)
    

    cv2.imshow('CannyEdge', result)
    cv2.waitKey(0)
    