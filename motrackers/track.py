import numpy as np
from motrackers.kalman_tracker import KFTracker2D, KFTrackerSORT, KFTracker4D


class Track:

    count = 0

    metadata = dict(
        data_output_formats=['mot_challenge', 'visdrone_challenge']
    )

    def __init__(
        self,
        track_id,
        frame_id,
        bbox,
        detection_confidence,
        class_id=None,
        lost=0,
        iou_score=0.,
        data_output_format='mot_challenge',
        **kwargs
    ):
        assert data_output_format in Track.metadata['data_output_formats']
        Track.count += 1
        self.id = track_id

        self.detection_confidence_max = 0.
        self.lost = 0
        self.age = 0

        self.update(frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs)

        if data_output_format == 'mot_challenge':
            self.output = self.get_mot_challenge_format
        elif data_output_format == 'visdrone_challenge':
            self.output = self.get_vis_drone_format
        else:
            raise NotImplementedError

    def update(self, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0., **kwargs):
        
        self.class_id = class_id
        self.bbox = np.array(bbox)
        self.detection_confidence = detection_confidence
        self.frame_id = frame_id
        self.iou_score = iou_score

        if lost == 0:
            self.lost = 0
        else:
            self.lost += lost

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.detection_confidence_max = max(self.detection_confidence_max, detection_confidence)

        self.age += 1

    @property
    def centroid(self):

        return np.array((self.bbox[0]+0.5*self.bbox[2], self.bbox[1]+0.5*self.bbox[3]))

    def get_mot_challenge_format(self):

        mot_tuple = (
            self.frame_id, self.id, self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], self.detection_confidence,
            -1, -1, -1
        )
        return mot_tuple

    def get_vis_drone_format(self):

        mot_tuple = (
            self.frame_id, self.id, self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3],
            self.detection_confidence, self.class_id, -1, -1
        )
        return mot_tuple

    def predict(self):
        """
        Implement to prediction the next estimate of track.
        """
        raise NotImplemented

    @staticmethod
    def print_all_track_output_formats():
        print(Track.metadata['data_output_formats'])


class KFTrackSORT(Track):

    def __init__(self, track_id, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0.,
                 data_output_format='mot_challenge', process_noise_scale=1.0, measurement_noise_scale=1.0, **kwargs):
        bbz = np.array([bbox[0]+0.5*bbox[2], bbox[1]+0.5*bbox[3], bbox[2]*bbox[3], bbox[2]/float(bbox[3])])
        self.kf = KFTrackerSORT(
            bbz, process_noise_scale=process_noise_scale, measurement_noise_scale=measurement_noise_scale)
        super().__init__(track_id, frame_id, bbox, detection_confidence, class_id=class_id, lost=lost,
                         iou_score=iou_score, data_output_format=data_output_format, **kwargs)

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        x = self.kf.predict()

        if x[2] * x[3] < 0:
            return np.array([np.nan, np.nan, np.nan, np.nan])

        w = np.sqrt(x[2] * x[3])
        h = x[2] / float(w)
        bb = np.array([x[0]-0.5*w, x[1]-0.5*h, w, h])
        return bb

    def update(self, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0., **kwargs):
        super().update(
            frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs)
        z = np.array([bbox[0]+0.5*bbox[2], bbox[1]+0.5*bbox[3], bbox[2]*bbox[3], bbox[2]/float(bbox[3])])
        self.kf.update(z)


class KFTrack4DSORT(Track):

    def __init__(self, track_id, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0.,
                 data_output_format='mot_challenge', process_noise_scale=1.0, measurement_noise_scale=1.0,
                 kf_time_step=1, **kwargs):
        self.kf = KFTracker4D(
            bbox.copy(), process_noise_scale=process_noise_scale, measurement_noise_scale=measurement_noise_scale,
            time_step=kf_time_step)
        super().__init__(track_id, frame_id, bbox, detection_confidence, class_id=class_id, lost=lost,
                         iou_score=iou_score, data_output_format=data_output_format, **kwargs)

    def predict(self):
        x = self.kf.predict()
        bb = np.array([x[0], x[3], x[6], x[9]])
        return bb

    def update(self, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0., **kwargs):
        super().update(
            frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs)
        self.kf.update(bbox.copy())


class KFTrackCentroid(Track):

    def __init__(self, track_id, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0.,
                 data_output_format='mot_challenge', process_noise_scale=1.0, measurement_noise_scale=1.0, **kwargs):
        c = np.array((bbox[0]+0.5*bbox[2], bbox[1]+0.5*bbox[3]))
        self.kf = KFTracker2D(c, process_noise_scale=process_noise_scale, measurement_noise_scale=measurement_noise_scale)
        super().__init__(track_id, frame_id, bbox, detection_confidence, class_id=class_id, lost=lost,
                         iou_score=iou_score, data_output_format=data_output_format, **kwargs)

    def predict(self):

        s = self.kf.predict()
        xmid, ymid = s[0], s[3]
        w, h = self.bbox[2], self.bbox[3]
        xmin = xmid - 0.5*w
        ymin = ymid - 0.5*h
        return np.array([xmin, ymin, w, h]).astype(int)

    def update(self, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0., **kwargs):
        super().update(
            frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs)
        self.kf.update(self.centroid)
