from collections import OrderedDict
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from motrackers.tracker import Tracker
from motrackers.track import KFTrackCentroid
from motrackers.utils.misc import get_centroid


def assign_tracks2detection_centroid_distances(bbox_tracks, bbox_detections, distance_threshold=10.):


    if (bbox_tracks.size == 0) or (bbox_detections.size == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(bbox_detections), dtype=int), np.empty((0,), dtype=int)

    if len(bbox_tracks.shape) == 1:
        bbox_tracks = bbox_tracks[None, :]

    if len(bbox_detections.shape) == 1:
        bbox_detections = bbox_detections[None, :]

    estimated_track_centroids = get_centroid(bbox_tracks)
    detection_centroids = get_centroid(bbox_detections)
    centroid_distances = distance.cdist(estimated_track_centroids, detection_centroids)

    assigned_tracks, assigned_detections = linear_sum_assignment(centroid_distances)

    unmatched_detections, unmatched_tracks = [], []

    for d in range(bbox_detections.shape[0]):
        if d not in assigned_detections:
            unmatched_detections.append(d)

    for t in range(bbox_tracks.shape[0]):
        if t not in assigned_tracks:
            unmatched_tracks.append(t)

    # filter out matched with high distance between centroids
    matches = []
    for t, d in zip(assigned_tracks, assigned_detections):
        if centroid_distances[t, d] > distance_threshold:
            unmatched_detections.append(d)
            unmatched_tracks.append(t)
        else:
            matches.append((t, d))

    if len(matches):
        matches = np.array(matches)
    else:
        matches = np.empty((0, 2), dtype=int)

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)


class CentroidKF_Tracker(Tracker):
    def __init__(
            self,
            max_lost=1,
            centroid_distance_threshold=30.,
            tracker_output_format='mot_challenge',
            process_noise_scale=1.0,
            measurement_noise_scale=1.0,
            time_step=1
    ):
        self.time_step = time_step
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale
        self.centroid_distance_threshold = centroid_distance_threshold
        self.kalman_trackers = OrderedDict()
        super().__init__(max_lost, tracker_output_format)

    def _add_track(self, frame_id, bbox, detection_confidence, class_id, **kwargs):
        self.tracks[self.next_track_id] = KFTrackCentroid(
            self.next_track_id, frame_id, bbox, detection_confidence, class_id=class_id,
            data_output_format=self.tracker_output_format, process_noise_scale=self.process_noise_scale,
            measurement_noise_scale=self.measurement_noise_scale, **kwargs
        )
        self.next_track_id += 1

    def update(self, bboxes, detection_scores, class_ids):
        self.frame_count += 1
        bbox_detections = np.array(bboxes, dtype='int')

        track_ids = list(self.tracks.keys())
        bbox_tracks = []
        for track_id in track_ids:
            bbox_tracks.append(self.tracks[track_id].predict())
        bbox_tracks = np.array(bbox_tracks)

        if len(bboxes) == 0:
            for i in range(len(bbox_tracks)):
                track_id = track_ids[i]
                bbox = bbox_tracks[i, :]
                confidence = self.tracks[track_id].detection_confidence
                cid = self.tracks[track_id].class_id
                self._update_track(track_id, self.frame_count, bbox, detection_confidence=confidence, class_id=cid, lost=1)
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)
        else:
            matches, unmatched_detections, unmatched_tracks = assign_tracks2detection_centroid_distances(
                bbox_tracks, bbox_detections, distance_threshold=self.centroid_distance_threshold
            )

            for i in range(matches.shape[0]):
                t, d = matches[i, :]
                track_id = track_ids[t]
                bbox = bboxes[d, :]
                cid = class_ids[d]
                confidence = detection_scores[d]
                self._update_track(track_id, self.frame_count, bbox, confidence, cid, lost=0)

            for d in unmatched_detections:
                bbox = bboxes[d, :]
                cid = class_ids[d]
                confidence = detection_scores[d]
                self._add_track(self.frame_count, bbox, confidence, cid)

            for t in unmatched_tracks:
                track_id = track_ids[t]
                bbox = bbox_tracks[t, :]
                confidence = self.tracks[track_id].detection_confidence
                cid = self.tracks[track_id].class_id
                self._update_track(track_id, self.frame_count, bbox, confidence, cid, lost=1)

                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

        outputs = self._get_tracks(self.tracks)
        return outputs
