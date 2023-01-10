import cv2
import numpy as np
import onnxruntime as ort
from numpy import random

from lib.helper import CWD
from lib.utils_numpy import (letterbox, non_max_suppression, plot_one_box,
                             scale_coords)


class ObjectDetector():
    """
        YoLo v7 Object Detector
    """
    def __init__(self) -> None:
        # Model Config
        self.session = None
        self.input_cfg = None
        self.input_name = None
        self.save_weights_path = CWD() + "/weight/onnx/yolo7.onnx"
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.stride = 32 # int(model.stride.max())  # model stride
        self.imgsz = 640
        self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        # PostProcess Config
        self.conf_thres = 0.20             # ObjectConfidenceThreshold
        self.iou_thres = 0.45              # NMS için IOU Threshold
        self.classes = None                # Sınıf tespiti için filtre 0-80: int veya [0,44]: array 
        self.agnostic_nms = False          # NMS için agnostic yöntemi

        # Sınıflar için rastgele renkler belirle
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        self.__StartNewSession()


    def __StartNewSession(self):
        self.session = ort.InferenceSession(self.save_weights_path, providers=self.providers)
        self.input_cfg = self.session.get_inputs()[0]
        self.input_name =  self.input_cfg.name


    def __Preprocess(self, frame):
        img = letterbox(frame, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = np.array(img, dtype=np.float16)
        img /= 255.0
        img = np.expand_dims(img, axis=0)

        return img


    def __Inference(self, input):
        pred = self.session.run(None, {self.input_name : input})[0]

        return pred


    def __PostProcess(self, pred, img, canvas, draw=True, verbose=False):
        """
            return: [top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence, class_number], canvas
        """
        detectedObjects = []

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # Batch olarak verilmişse çıkarımları tek tek yap
            s = ''

            if len(det):
                # Orijinal resim için koordinatları yeniden boyutlandır.
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], canvas.shape).round()

                if verbose:
                    for c in np.unique(det[:, -1]):
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, " #Objenin isim ve puanını ekle
                        print(s)

                # Sonuçları canvasa bastır.
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    detectedObjects.append([*xyxy, conf, cls])
                    if draw:
                        canvas = plot_one_box(xyxy, canvas, label=label, color=self.colors[int(cls)])

        return detectedObjects, canvas


    def Detect(self, frame, draw=True):
        canvas = frame.copy()
        frame = self.__Preprocess(frame)
        pred = self.__Inference(frame)
        detectedObjects, canvas = self.__PostProcess(pred, frame, canvas, draw=draw)

        return detectedObjects, canvas



if __name__ == '__main__':
    objectDetector = ObjectDetector()

    cam = cv2.VideoCapture("asset/V01.mp4")

    while(True):
        ret, frame = cam.read()
        if not ret: break

        frame = cv2.resize(frame, (1280, 720))
        detectedObjects, canvas  = objectDetector.Detect(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"): break
        cv2.imshow('', canvas)
        