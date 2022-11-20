def align_plate(image):
    nonZeroCoordinates = cv2.findNonZero(image)
    imageCopy = image.copy()
    for pt in nonZeroCoordinates:
        imageCopy = cv2.circle(imageCopy, (pt[0][0], pt[0][1]), 1, (255, 0, 0))
    showimg("imageCopy", imageCopy)
    
    box = cv2.minAreaRect(nonZeroCoordinates)
    boxPts = cv2.boxPoints(box)
    print(boxPts)
    for i in range(4):
        pt1 = (boxPts[i][0], boxPts[i][1])
        pt2 = (boxPts[(i+1)%4][0], boxPts[(i+1)%4][1])
        print(pt1, pt2)
        cv2.line(imageCopy, pt1, pt2, (0,255,0), 2, cv2.LINE_AA)
    angle = box[2]
    if(angle < -45):
        angle = 90 + angle
   
    h, w, c = image.shape
    scale = 1.
    center = (w/2., h/2.)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = image.copy()
    cv2.warpAffine(image, M, (w, h), rotated, cv2.INTER_CUBIC, cv2.BORDER_REPLICATE )
    showimg("warped", image)
