import numpy as np
import cv2 as cv
from motrackers.utils.misc import draw_tracks
from motrackers.utils.misc import xyxy2xywh
from PIL import Image, ImageFilter
import os

class Detector:

    def __init__(self, object_names, confidence_threshold, nms_threshold, draw_bboxes=True):
        self.object_names = object_names
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.height = None
        self.width = None

        np.random.seed(12345)
        if draw_bboxes:
            self.bbox_colors = {key: np.random.randint(0, 255, size=(3,)).tolist() for key in self.object_names.keys()}
       # print(self.bbox_colors)

    def forward(self, image):
        raise NotImplemented

    def detect(self, image):
        if self.width is None or self.height is None:
            (self.height, self.width) = image.shape[:2]

        detections = self.forward(image).squeeze(axis=0).squeeze(axis=0)
        bboxes, confidences, class_ids = [], [], []

        for i in range(detections.shape[0]):
            detection = detections[i, :]
            class_id = detection[1]
            confidence = detection[2]

            if confidence > self.confidence_threshold:
                bbox = detection[3:7] * np.array([self.width, self.height, self.width, self.height])
                bboxes.append(bbox.astype("int"))
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

        if len(bboxes):
            bboxes = xyxy2xywh(np.array(bboxes)).tolist()
            class_ids = np.array(class_ids).astype('int')
            indices = cv.dnn.NMSBoxes(bboxes, confidences, self.confidence_threshold, self.nms_threshold).flatten()
            return np.array(bboxes)[indices, :], np.array(confidences)[indices], class_ids[indices]
        else:
            return np.array([]), np.array([]), np.array([])

    def draw_bboxes(self, image, bboxes, confidences, class_ids, tracks, colorTracker):

        # image1 = Image.fromarray(image)
        # image1 = image1.filter(ImageFilter.GaussianBlur)
        for bb, conf, cid, trk in zip(bboxes, confidences, class_ids, tracks):
            trk_id = trk[1]
            # try:
            #     trk_dict[trk_id] = self.object_names[cid]
            # except KeyError:
            #     trk_dict[trk_id] = ' '
            if trk_id not in colorTracker.keys():
                colorTracker[trk_id] = np.random.randint(0, 255, size=(3,)).tolist()
            
            #cv.rectangle(image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), clr, 2)
            #print(self.object_names)
            label = self.object_names[cid]
            #print(label)
            (label_width, label_height), baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(bb[1]-10, label_height)
            cv.rectangle(image, (bb[0], y_label - label_height), (bb[0] + label_width, y_label + baseLine),
                         (255, 255, 255), cv.FILLED)
            cv.putText(image, label, (bb[0], y_label), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # image1 = Image.fromarray(image)
        updated_image = draw_tracks(image, tracks, colorTracker)
        return updated_image, colorTracker

    def draw_bboxes2(self, image, bboxes, confidences, class_ids):

        # image1 = Image.fromarray(image)
        # image1 = image1.filter(ImageFilter.GaussianBlur)
        for bb, conf, cid in zip(bboxes, confidences, class_ids):
            clr = [int(c) for c in self.bbox_colors[cid]]
            cv.rectangle(image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), clr, 2)
            try:
                label = "{}".format(self.object_names[cid])
            except KeyError:
                continue
            #print(label)
            (label_width, label_height), baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(bb[1], label_height)
            cv.rectangle(image, (bb[0], y_label - label_height), (bb[0] + label_width, y_label + baseLine),
                         (255, 255, 255), cv.FILLED)
            cv.putText(image, label, (bb[0], y_label), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
            # image1 = Image.fromarray(image)
        return image

    def checkPerson(self, cid):
        if (self.object_names[cid] == 'person'):
            return True
        else:
            return False

    def checkCar(self, cid):
        if (self.object_names[cid] == 'car' or self.object_names[cid] == 'motorbike'):
            return True
        else:
            return False

    def convertBack(self, x, y, w, h): 
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax
