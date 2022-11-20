img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Blackhat
rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (81,3))
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, rectKern)#Closing(Dilation+Erosion) #reveal dark characters 
showimg("A1-blackhat", blackhat)

#Closing
squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, squareKern)
showimg("B1-closing", closing)

#Threshold_OTSU
closing_thres = cv2.threshold(closing, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
showimg("B2-threshold", closing_thres)
return closing_thres
"""
#Sobel Kenar Çıkartma
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
showimg("A2-Sobel", gradX)

#Normalizasyon
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = 255 * ((gradX - minVal) / (maxVal - minVal) + sys.float_info.epsilon) #0 a bölme hatası olmaması için "sys.float_info.epsilon"
gradX = np.array(gradX, np.uint8)
showimg("A3-Normalizasyon", gradX)

#Blur
gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
showimg("A4-Blur", gradX)

#Closing
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
showimg("A5-Closing", gradX)

#Threshold_OTSU
thresh = cv2.threshold(gradX, 0, 255,cv2. THRESH_BINARY | cv2.THRESH_OTSU)[1]
showimg("A6-thres|thres_otsu", thresh)

thresh = cv2.erode(thresh, None, iterations=1)
thresh = cv2.dilate(thresh, None, iterations=1)
showimg("A7-erode|dilate", thresh)

thresh = cv2.bitwise_and(thresh, thresh, mask=closing_thres)
showimg("A8-bitwise_and(B2-threshold)", thresh)

thresh = cv2.dilate(thresh, None, iterations=2)
thresh = cv2.erode(thresh, None, iterations=1)
showimg("A9-dilate2_erode", thresh)
    
