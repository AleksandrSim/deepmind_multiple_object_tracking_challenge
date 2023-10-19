import os
import time
import queue
import colorsys
import threading

import cv2
import numpy as np

MAX_QUEUE_SIZE = 2000
SLEEP_TIME = 0.01

FONT_SIZE = 1.5
FONT_THIKNESS = int(round(FONT_SIZE))

COL_WHITE = (255, 255, 255)

INFO_TEXT_COLOR = (0, 255, 0)
DETECT_BBOX_COLOR = (255, 0, 0)
GT_BBOX_COLOR = (0, 0, 0)

FRAMES_TO_KEEP_GT = 20


def create_unique_color(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Args:
        tag (int): The unique target identifying tag.
        hue_step (float): Difference between two neighboring color codes in
        HSV space (more specifically, the distance in hue channel).

    Returns:
        (int, int, int): RGB color code in range

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return int(r*255), int(g*255), int(b*255)


class Visualizer:
    def __init__(self, video_path: str, output_video_path: str):
        assert os.path.exists(video_path), f'Could not find {video_path}'
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.video_writer = cv2.VideoWriter(output_video_path,
                                            cv2.VideoWriter_fourcc(*'mp4v'),
                                            fps, (frame_width, frame_height))
        self.data_queue = queue.Queue(MAX_QUEUE_SIZE)
        self.stopped = False
        self.thread = None
        self.gts: dict[int, dict] = {}

    def draw(self, frame_idx, detections, tracks):
        if frame_idx is not None:
            det_res = [(det.to_tlbr(), det.track_id) for det in detections]
            track_res = [(track.to_tlbr(), track.track_id) for track in tracks
                         if (track.is_confirmed()
                             and track.time_since_update <= 0)]
            self.data_queue.put((frame_idx, det_res, track_res))
        else:
            self.data_queue.put(None)

    def _draw_text(self, frame, text: str, x, y,
                   color,
                   background_color= None,
                   alpha: float = 1.0):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN,
                                    FONT_SIZE, FONT_THIKNESS)
        center = (int(x + 10*FONT_SIZE/2),
                  int(y + 10*FONT_SIZE/2 + text_size[0][1]))
        if background_color is None:
            cv2.putText(frame, text, center, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, color, FONT_THIKNESS)
        else:
            h, w, _ = frame.shape
            x = min(max(0, x), w)
            y = min(max(0, y), h)
            pt2 = (min(max(0, int(x + 10*FONT_SIZE + text_size[0][0])), w),
                   min(max(0, int(y + 10*FONT_SIZE + text_size[0][1])), h))

            patch = np.ones((pt2[1]-y, pt2[0]-x, 3), dtype=np.uint8)
            patch[:, :] = background_color
            cv2.putText(patch, text, (int(10*FONT_SIZE/2), int(15*FONT_SIZE)),
                        cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, color, FONT_THIKNESS)
            if abs(y-pt2[1]) > 0 and abs(x-pt2[0]) > 0:
                frame[y:pt2[1], x:pt2[0], :] = cv2.addWeighted(
                    frame[y:pt2[1], x:pt2[0], :], 1 - alpha, patch, alpha, 0.0)
        return frame

    def _update_gts(self, detection, frame_idx: int):
        bbox, track_id = detection
        self.gts[track_id] = {'bbox': bbox, 'frame_idx': frame_idx}

    def _draw_bbox(self, frame, pos, label=None,
                   color=COL_WHITE,
                   crossed= False):
        x1, y1, x2, y2 = map(int, pos)
        try:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            if crossed:
                cv2.line(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
                cv2.line(frame, (x2, y1), (x1, y2), color, 1, cv2.LINE_AA)
            if label is not None:
                self._draw_text(frame, label, x1, y1, COL_WHITE, color, 0.5)
            return frame
        except:
            print('Cannot visualize', x1, y1, x2, y2)

    def _visualize(self, frame: np.ndarray, det_res,
                   track_res, frame_idx):
        frame = self._draw_text(frame, str(frame_idx), 10, 20, INFO_TEXT_COLOR)
        for det in det_res:
            if det[1] >= 0:
                self._update_gts(det, frame_idx)
            else:
                frame = self._draw_bbox(frame, det[0], color=DETECT_BBOX_COLOR)
        for track_id, gt in self.gts.items():
            if frame_idx - gt['frame_idx'] < FRAMES_TO_KEEP_GT:
                frame = self._draw_bbox(frame, gt['bbox'], label=str(track_id),
                                        color=GT_BBOX_COLOR, crossed=True)
        for track in track_res:
            bbox, track_id = track
            frame = self._draw_bbox(frame, bbox, label=str(track_id),
                                    color=create_unique_color(track_id))
        return frame

    def _vis_thread(self):
        print('Start vis thread')
        while not self.stopped:
            if self.data_queue.qsize() > 0:
                sample = self.data_queue.get()
                if sample is None:
                    self.stopped = True
                    break
                frame_idx, det_res, track_res = sample
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                if ret:
                    frame = self._visualize(frame=frame, det_res=det_res,
                                            track_res=track_res,
                                            frame_idx=frame_idx)
                    self.video_writer.write(frame)
            time.sleep(SLEEP_TIME)

    def start(self):
        self.thread = threading.Thread(target=self._vis_thread)
        self.thread.start()

    def release(self):
        if self.thread is not None:
            self.thread.join()
        self.cap.release()
        self.video_writer.release()