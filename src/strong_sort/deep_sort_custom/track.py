# vim: expandtab:ts=4:sw=4
import yaml
import numpy as np

from .detection import Detection
from .kalman_filter import KalmanFilter

with open("config.yml", 'r') as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    detection (Detection): The initial detection.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    detection : Detection) The initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, detection: Detection, track_id: int, n_init: int,
                 max_age: int, gt: bool = False):
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if detection.feature is not None:
            self.features.append(detection.feature)

        self.scores = []
        if detection.confidence is not None:
            self.scores.append(detection.confidence)
        self.gt = gt

        self._n_init = n_init
        self._max_age = max_age

        self._init_state(detection)

    def _init_state(self, detection: Detection):
        self.kf: KalmanFilter = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(detection.to_xyah())
        self.mean_prev, self.covariance_prev = self.mean, self.covariance

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        """
        self.mean_prev, self.covariance_prev = self.mean, self.covariance
        self.mean, self.covariance = self.kf.predict(
            self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    @staticmethod
    def get_matrix(dict_frame_matrix, frame):
        eye = np.eye(3)
        matrix = dict_frame_matrix[frame]
        dist = np.linalg.norm(eye - matrix)
        if dist < 100:
            return matrix
        else:
            return eye

    def camera_update(self, video, frame):
        dict_frame_matrix = cfg['arguments']['ECC']
        frame = str(int(frame))
        if frame in dict_frame_matrix:
            matrix = self.get_matrix(dict_frame_matrix, frame)
            x1, y1, x2, y2 = self.to_tlbr()
            x1_, y1_, _ = matrix @ np.array([x1, y1, 1]).T
            x2_, y2_, _ = matrix @ np.array([x2, y2, 1]).T
            w, h = x2_ - x1_, y2_ - y1_
            cx, cy = x1_ + w / 2, y1_ + h / 2
            self.mean[:4] = [cx, cy, w / h, h]

    def _update_feature(self, feature):
        if cfg['arguments']['EMA']:
            smooth_feat = cfg['arguments']['EMA_alpha'] * \
                self.features[-1] + (1 - cfg['arguments']
                                     ['EMA_alpha']) * feature
            smooth_feat /= np.linalg.norm(smooth_feat)
            self.features = [smooth_feat]
        else:
            self.features.append(feature)

    def update(self, detection: Detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, detection.to_xyah(), detection.confidence)
        self._update_feature(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def update_force(self, detection):
        self._init_state(detection)
        self._update_feature(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def update_constant(self):
        self.mean, self.covariance = self.mean_prev, self.covariance_prev
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
            print(f"Marking missed for tentative track ID: {self.track_id}")

        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
            print(f"Marking missed by age for track ID: {self.track_id}")

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
