import numpy as np
import cv2 as cv
from motrackers.detectors.detector import Detector
from motrackers.utils.misc import load_labelsjson


class YOLOv3(Detector):

    def __init__(self, weights_path, configfile_path, labels_path, confidence_threshold=0.2, nms_threshold=0.2, draw_bboxes=True, use_gpu=False):
        self.net = cv.dnn.readNetFromDarknet(configfile_path, weights_path)
        object_names = load_labelsjson(labels_path)

        layer_names = self.net.getLayerNames()
        self.layer_names = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.scale_factor = 1/255.0
        self.image_size = (416, 416)

        self.net = cv.dnn.readNetFromDarknet(configfile_path, weights_path)
        if use_gpu:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

        super().__init__(object_names, confidence_threshold, nms_threshold, draw_bboxes)

    def forward(self, image):
        blob = cv.dnn.blobFromImage(image, self.scale_factor, self.image_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward(self.layer_names)  # detect objects using object detection model
        return detections

    def detect(self, image):
        if self.width is None or self.height is None:
            (self.height, self.width) = image.shape[:2]

        detections = self.forward(image)

        bboxes, confidences, class_ids = [], [], []

        for output in detections:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                #print(confidence)
                if confidence > self.confidence_threshold:
                    xmid, ymid, w, h = detect[0:4] * np.array([self.width, self.height, self.width, self.height])
                    x, y = int(xmid - 0.5*w), int(ymid - 0.5*h)
                    bboxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        if len(cv.dnn.NMSBoxes(bboxes, confidences, self.confidence_threshold, self.nms_threshold)) > 0:
            indices = cv.dnn.NMSBoxes(bboxes, confidences, self.confidence_threshold, self.nms_threshold).flatten()
            class_ids = np.array(class_ids).astype('int')
            output = np.array(bboxes)[indices, :].astype('int'), np.array(confidences)[indices], class_ids[indices]
            return True, output
        else:
            return False, image 
