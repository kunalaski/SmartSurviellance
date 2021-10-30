from warnings import filterwarnings
filterwarnings('ignore')
import pandas as pd
import numpy as np
import csv
import time
import os

import cv2 as cv
from motrackers.detectors import YOLOv3
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
#from motrackers.utils import draw_tracks
import ipywidgets as widgets

VIDEO_FILE = "test.mp4"
WEIGHTS_PATH = './examples/pretrained_models/yolo_weights/yolov4-tiny.weights'
CONFIG_FILE_PATH = './examples/pretrained_models/yolo_weights/yolov4-tiny.cfg'
LABELS_PATH = "./examples/pretrained_models/yolo_weights/coco_names.json"
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 5

MAX_SEQ_LENGTH = 2000
NUM_FEATURES = 2048
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.2
DRAW_BOUNDING_BOXES = True
USE_GPU = True

tracker = CentroidKF_Tracker(max_lost=0, tracker_output_format='mot_challenge')

model = YOLOv3(
    weights_path=WEIGHTS_PATH,
    configfile_path=CONFIG_FILE_PATH,
    labels_path=LABELS_PATH,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    nms_threshold=NMS_THRESHOLD,
    draw_bboxes=DRAW_BOUNDING_BOXES,
    use_gpu=USE_GPU
)

def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image

def main(video_path, model, tracker):
    frames = []
    cap = cv.VideoCapture(video_path)
    colorTracker = {}
    trk_dict={}
    count = 0
    prev_frame_time = 0
    new_frame_time = 0
    try:
        while True:
            ok, image = cap.read()

            if not ok:
                print("Cannot read the video feed.")
                break
            
            image = cv.resize(image, (700, 500))
            booli, output = model.detect(image)
            if(booli != False):
                bboxes, confidences, class_ids = output
                tracks = tracker.update(bboxes, confidences, class_ids)
                #print(class_ids)
    #             print(len(tracks))
                image, colorTracker = model.draw_bboxes(image.copy(), bboxes, confidences, class_ids, tracks, colorTracker)
    #             print(updated_image)
                
                #frame = crop_center_square(updated_image)
#             gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
#             _,thresh = cv.threshold(gray,1,255,cv.THRESH_BINARY)
#             contours,hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
#             cnt = contours[0]
#             x,y,w,h = cv.boundingRect(cnt)
#             frame = image[y:y+h,x:x+w]
            #print(image.shape)
            image = autocrop(image)
            frame = cv.resize(image, (224, 224))
            frame = frame[:, :, [2, 1, 0]]
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
           # frame = frame[20:210, 10:210]
            frames.append(frame)
            count += 1
            #print(count)
            #print(type(frames))
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            #print(f'{count}-{fps}')
            cv.imshow("image", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()
    return np.array(frames)
main('D:\kunal\Project\smart surveillance\ModelCode\TrainTestData\Anomaly-Videos-Part-1\Abuse\Abuse039_x264.mp4', model, tracker)