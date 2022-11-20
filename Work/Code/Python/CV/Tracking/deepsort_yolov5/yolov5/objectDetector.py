import math
import os
import time

import cv2
import numpy as np
import onnxruntime
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True  # Set True to speed up constant image size inference
import tensorflow as tf
import torchvision


class ObjectDetector():
    def __init__(self, conf_thres = 0.35, iou_thres = 0.45, max_det = 1000, classes = [0]):
        self.new_img_size = [640]
        self.new_img_size *= 2 if len(self.new_img_size) == 1 else 1
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.hide_conf = False
        
        self.model_root = os.getcwd() + "/weights/yolov5l.onnx"
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  #'TensorrtExecutionProvider'
        self.session = onnxruntime.InferenceSession(self.model_root, providers=self.providers)
        self.input_cfg = self.session.get_inputs()[0]
        self.input_name = self.input_cfg.name
        self.outputs = self.session.get_outputs()
        self.output_names = [o.name for o in self.outputs]

        self.stride = 32
        self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def check_img_size(self, new_img_size, s=32):
        s = int(s)
        new_size = math.ceil(new_img_size / s) * s
        return new_size

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300):
        """
        Non-Maximum Suppression (NMS): Çakışan boxları temizleyerek en doğru olanı yazdırır.

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        bs = prediction.shape[0]  # batch size : input output toplam resim sayısı
        nc = prediction.shape[2] - 5  # class sayısı: 85-5 = 80 label var.
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.1 + 0.03 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break  # time limit exceeded

        return output

    def scale_coords(self, old_size, coords, new_size, ratio_pad=None):
        if ratio_pad is None:  # calculate from new_size
            gain = min(old_size[0] / new_size[0], old_size[1] / new_size[1])
            pad = (old_size[1] - new_size[1] * gain) / 2, (old_size[0] - new_size[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain

        # Taşan değerlerin mutlaklarını al
        self.clip_coords(coords, new_size)
        return coords

    def clip_coords(self, boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # Tek başına daha hızlı
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
        
        shape = im.shape[:2]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        #Only Scale Down
        if not scaleup:
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # w-h padding
        
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        # Resme kenarlık eklemek için w-h değerlerini 2 ye böl ve resmin dört tarafına kenarlık ekleneceğini belirt
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # Border Ekle
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (dw, dh)

    def drawPredictions(self, canvas_img, bboxes, classIds, scores):
        
        for np_bbox, class_id, class_score in zip(bboxes, classIds, scores):
            label = f'{self.names[class_id]} {class_score:.2f}'
            color = (128, 128, 128)
            txt_color=(255, 255, 255)

            lw = 3 or max(round(sum(canvas_img.shape) / 2 * 0.003), 2)  # line width
            p1, p2 = (np_bbox[:2]), (np_bbox[2:])

            canvas_img = cv2.rectangle(canvas_img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
            
            if label:
                tf = max(lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = [p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3]
                
                canvas_img = cv2.putText(canvas_img, label, (p1[0],
                    p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3,
                    txt_color,thickness=tf,lineType=cv2.LINE_AA
                )

        return canvas_img

    def detect(self, image, draw=False):
        image = image[..., ::-1]  # BGR to RGB, BHWC to BCHW
        canvas_img =  image.copy()
        ORIGINAL_IMAGE_SHAPE = image.shape
        self.new_img_size = self.check_img_size(new_img_size=640, s=self.stride)
        
        # Convert
        cropped_img, _, _ = self.letterbox(image)
        cropped_img = np.expand_dims(cropped_img, axis=0)
        cropped_img = cropped_img.transpose((0, 3, 1, 2))
        CROPPED_IMAGE_SHAPE = cropped_img.shape[2:]

        # Optimization and normalization
        image = np.ascontiguousarray(cropped_img)
        image = np.squeeze(np.float16(image)/255)

        # Inference
        pred = self.session.run(None, {self.input_name : [image]})

        # TODO Tensorflow NMS ye çevrilecek, çünkü why not tensorflow > pytorch :)
        """
        selected_indices = tf.image.non_max_suppression(boxes, scores, output, iou_threshold=0.45)
        selected_boxes = tf.gather(boxes, selected_indices)
        """

        # NMS 
        pred = np.array(pred)
        pred = torch.FloatTensor(pred).cuda()
        pred = self.non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, agnostic=True, max_det=self.max_det)
        
        bboxes = np.array([[0,0,0,0]], np.int32)
        scores = np.array([], dtype=np.float32)
        classIds = np.array([], np.int32)
        
        for i, pred in enumerate(pred):
            if len(pred):
                
                # Rescale boxes
                pred[:, :4] = self.scale_coords(CROPPED_IMAGE_SHAPE, pred[:, :4], ORIGINAL_IMAGE_SHAPE).round()

                for *box, class_score, cls in reversed(pred.detach().cpu().numpy()):
                    np_bbox = np.array([[ box[0], box[1], box[2], box[3] ]], np.int32)
                    np_score = float(class_score)
                    np_classId = int(cls)
                    
                    bboxes = np.append(bboxes, np_bbox, axis=0)
                    scores = np.append(scores, np_score)
                    classIds = np.append(classIds, np_classId)

                    #Draw Boxes
        if draw:
            canvas_img = self.drawPredictions(canvas_img, bboxes, classIds, scores)

        return bboxes[1:], scores, classIds, canvas_img


if __name__ == "__main__":
    detector = ObjectDetector()
    cam = cv2.VideoCapture("crowd.mp4")
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




