import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import cv2
import matplotlib.pyplot as plt
import numpy as np

#DETECTOR
from yolov5.objectDetector import ObjectDetector

#DEEPSORT
from deep_sort import nn_matching, preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.generate_detections import create_box_encoder

class DeepSORT(object):
    def __init__(self, objectDetector=None, iou = 0.45, score = 0.5, max_cosine_distance = 0.4, nms_max_overlap = 1.0):
        
        #NMS
        self.iou = iou   #0.45
        self.score = score  #5
        self.nms_max_overlap = nms_max_overlap #1.0

        self.max_cosine_distance = max_cosine_distance #0.4
        self.nn_budget = None

        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        self.counted_object_info = True
        # Initialize DeepSORT
        self.model_filename = 'weights/mars-small128.pb'
        self.encoder = create_box_encoder(self.model_filename, batch_size=1)

        # Initialize Tracker 
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)


    def DrawBoxes(self, frame, bbox, track, colors, class_name, info=False):
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        if info:
            print(f"\
                o-> Tracker ID: {str(track.track_id)},\n \
                o-> Class: {class_name},\n \
                o-> BBox Coords (xmin, ymin, xmax, ymax): {(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))}\n \
            ")
        return frame


    def Track(self, frame, bboxes=[], scores=[], classes=[], draw=False):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cFrame = frame.copy()
        

        num_objects = len(classes)
        # Filter Classes
        allowed_classes = ['person']

        # Filter Results
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = self.class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        if self.counted_object_info:
            cFrame = cv2.putText(cFrame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        
        # Delete boxes except FilterNames
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # Feed Tracker with encoded frame which is extracted from yolov5
        features = self.encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        """
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       
        """

        # Start Tracker
        self.tracker.predict()
        self.tracker.update(detections)

        person_bboxes = []
        # Update Tracks
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            
            bbox = track.to_tlbr()
            person_bboxes.append(bbox)

            class_name = track.get_class()

            if draw:
                cFrame = self.DrawBoxes(cFrame, bbox, track, colors, class_name, info=False)
                
        #Convert Image
        cFrame = np.asarray(cFrame)
        cFrame = cv2.cvtColor(cFrame, cv2.COLOR_BGR2RGB)

        return person_bboxes, cFrame


if __name__ == "__main__":
    detector = ObjectDetector()
    deepTracker = DeepSORT()

    cam = cv2.VideoCapture("videos/crowd.mp4")

    while True:
        rez, frame = cam.read()
        
        if rez:
            bboxes, scores, classIds, canvas = detector.detect(frame, draw=True)
            person_bboxes, cFrame = deepTracker.Track(frame, bboxes=bboxes, scores=scores, classes=classIds, draw=True)

        cv2.imshow("Tracker Test", cFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
