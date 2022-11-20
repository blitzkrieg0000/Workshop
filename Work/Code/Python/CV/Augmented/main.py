import cv2
import numpy as np

class FeatureExtractor(object):
    def __init__(self) -> None:
        self.ORBFeatureExtractor = cv2.ORB_create(nfeatures=1000)
        self.BruteForceMatcher = cv2.BFMatcher()                    #cv2.NORM_HAMMING, crossCheck=True
        self.targetKeypoints = None
        self.__targetFrame = None
        self.targetFrame_h=None
        self.targetFrame_w=None
        self.targetFrame_c=None
        self.__targetKeypoints = None
        self.__targetDescriptors = None
        self.__canvasFrame = None
        self.GOOD_POINT_THRESHOLD = 40
        self.GOOD_POINT_DISTANCE_MULTIPLIER = 0.73
    @property
    def targetFrame(self):
        return self.__targetFrame

    @targetFrame.setter
    def targetFrame(self, val):
        self.__targetFrame = val

    @property
    def canvasFrame(self):
        return self.__canvasFrame

    @canvasFrame.setter
    def canvasFrame(self, frame):
        self.__canvasFrame = frame

    @property
    def targetKeypoints(self):
        return self.__targetKeypoints

    @targetKeypoints.setter
    def targetKeypoints(self, val):
        self.__targetKeypoints = val

    @property
    def targetDescriptors(self):
        return self.__targetDescriptors
    
    @targetDescriptors.setter
    def targetDescriptors(self, val):
        self.__targetDescriptors = val


    def SetTargetFrame(self, frame):
        self.targetFrame = frame
        self.targetFrame_h, self.targetFrame_w, self.targetFrame_c = self.targetFrame.shape
        if self.targetFrame is not None:
            self.targetKeypoints, self.targetDescriptors = self.ExtractKeyPoints(self.targetFrame)


    def SetCanvasFrame(self, frame):
        self.canvasFrame = frame


    def ExtractKeyPoints(self, frame):
        keypoints, descriptors = self.ORBFeatureExtractor.detectAndCompute(frame, None)
        return keypoints, descriptors
    

    def GetMatchesByKNN(self, source_descriptors):
        if source_descriptors is None:
            return []
        
        goodPoints  = []
        matches = self.BruteForceMatcher.knnMatch(self.__targetDescriptors, source_descriptors, k=2)

        if len(matches)> 0:
            if len(matches[0])>1:
                for m, n in matches:
                    if m.distance < self.GOOD_POINT_DISTANCE_MULTIPLIER * n.distance:
                        goodPoints.append(m)
                        
        return goodPoints


    def FindHomographyPerspective(self, frame, source_keypoints, good_points):
        
        if len(good_points) > self.GOOD_POINT_THRESHOLD:
            sourcePoints = np.float32([self.targetKeypoints[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            destinationPoints = np.float32([source_keypoints[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(sourcePoints, destinationPoints,cv2.RANSAC, 5)
            
            pts = np.float32([[0,0], [0, self.targetFrame_h], [self.targetFrame_w, self.targetFrame_h], [self.targetFrame_w,0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
    
            #img2 = cv2.polylines(frame, [np.int32(dst)], True, (255,0,255), 3)

            imgWarp = cv2.warpPerspective(FE.canvasFrame, matrix, (FE.canvasFrame.shape[1], FE.canvasFrame.shape[0]))
            maskNew = np.zeros((FE.canvasFrame.shape[0], FE.canvasFrame.shape[1]), np.uint8)
            maskNew = cv2.fillPoly(maskNew, [np.int32(dst)], (255,255,255))
            maskInv = cv2.bitwise_not(maskNew)
            imgAug = frame.copy()
            imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
            imgAug = cv2.bitwise_or(imgWarp , imgAug)
            return imgAug


    def DrawMatches(self, source_keypoints, frame, good_points):
        return cv2.drawMatches(self.targetFrame, self.targetKeypoints, frame, source_keypoints, good_points, None, flags=2)    


    def detect(self, source):
        source = cv2.resize(source, (self.targetFrame_w, self.targetFrame_h))


def GetSources():
    cam = cv2.VideoCapture(0)
    source = cv2.imread("test1.jpg")
    target = cv2.imread("test2.jpg")
    target = cv2.resize(target, (640, 480))
    return cam, target, source


if '__main__' == __name__:

    FE = FeatureExtractor()

    #! GET SOURCES
    cam, targetImage, source = GetSources()
    FE.SetTargetFrame(targetImage)

    source = cv2.resize(source, (FE.targetFrame_w, FE.targetFrame_h))
    FE.SetCanvasFrame(source)


    while True:
        ret, frame = cam.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            keypoints, descriptors = FE.ExtractKeyPoints(frame)
            goodpoints = FE.GetMatchesByKNN(descriptors)
            canvas = FE.FindHomographyPerspective(frame, keypoints, goodpoints)

            cv2.imshow('', canvas if canvas is not None else frame)
            cv2.waitKey(1)
            


















