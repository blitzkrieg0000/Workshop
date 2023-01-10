import cv2
import numpy as np
import onnxruntime as ort
from numpy import random

from lib.utils import (letterbox, non_max_suppression, plot_one_box, scale_coords)


def detect():
    # ONNX
    save_weights_path = "weight/yolo7.onnx"
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(save_weights_path, providers=providers)
    input_cfg = session.get_inputs()[0]
    input_name = input_cfg.name

    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    stride = 32 # int(model.stride.max())  # model stride
    imgsz = 640

    conf_thres = 0.20             # object confidence threshold
    iou_thres = 0.45              # IOU threshold for NMS
    classes = None                # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False


    # Get names and colors
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    cam = cv2.VideoCapture("asset/V01.mp4")

    while(True):
        ret, frame = cam.read()

        if not ret:
            break

        canvas = frame.copy()
        
        img = letterbox(frame, imgsz, stride=stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = np.array(img, dtype=np.float16)
        img /= 255.0
        img = np.expand_dims(img, axis=0)

        #Inference
        pred = session.run(None, {input_name : img})[0]
        #pred = torch.tensor(pred)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # Batch olarak verilmişse çıkarımları tek tek yap
            s = ''

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], canvas.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, canvas, label=label, color=colors[int(cls)], line_thickness=1)

                # Show
                cv2.imshow("", cv2.resize(canvas,(1280, 720)))
                cv2.waitKey(1)



if __name__ == '__main__':
    detect()


