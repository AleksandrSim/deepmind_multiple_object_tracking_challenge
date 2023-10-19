import cv2
import os

def write_frames(video_path, output):
    cap = cv2.VideoCapture(video_path)
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 640))  # Changed to a tuple
        cv2.imwrite(os.path.join(output, f'{idx}.png'), frame)  # Added the frame
        idx += 1

if __name__=='__main__':  # Corrected the condition
    write_frames('/Users/aleksandrsimonyan/Desktop/deepmind/videos/video_1.mp4', # Make sure to have a leading slash
                 '/Users/aleksandrsimonyan/Desktop/yolo_weights/frames')