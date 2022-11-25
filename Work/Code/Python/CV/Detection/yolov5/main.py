import cv2
from lib.objectDetector import YoLoObjectDetector


if __name__ == "__main__":
    detector = YoLoObjectDetector(conf_thres = 0.35, iou_thres = 0.45, max_det = 1000, classes = [0, 32])
    cam = cv2.VideoCapture("assets/video/crowd.mp4")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        success, img = cam.read()

        if not success:
            break
        
        #PROCESS
        bboxes, scores, classIds, canvas = detector.detect(img, draw=True)

        for b,s,c in zip(bboxes, scores, classIds):
            print(b, s, c, sep=" <-> ")

        cv2.imshow("", canvas)
        cv2.waitKey(1)