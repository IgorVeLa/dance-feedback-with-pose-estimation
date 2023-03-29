import enum
from PIL import Image
from pathlib import Path
import os


class PoseLandmark(enum.IntEnum):
    """Overwrite the MediaPipes 33 pose landmarks into 16."""
    NOSE = 0
    LEFT_SHOULDER = 1
    RIGHT_SHOULDER = 2
    LEFT_ELBOW = 3
    RIGHT_ELBOW = 4
    LEFT_WRIST = 5
    RIGHT_WRIST = 6
    LEFT_HIP = 7
    RIGHT_HIP = 8
    LEFT_KNEE = 9
    RIGHT_KNEE = 10
    LEFT_ANKLE = 11
    RIGHT_ANKLE = 12
    LEFT_HEEL = 13
    RIGHT_HEEL = 14
    LEFT_FOOT_INDEX = 15
    RIGHT_FOOT_INDEX = 16


POSE_CONNECTIONS = frozenset([(1, 2), (1, 3), (1, 7), (3, 5), # Left upper
                              (2, 4), (2, 8), (4, 6), # right upper
                              (7, 8), (7, 9), (9, 11), (11, 13), (13, 15), # left lower
                              (8, 10), (10, 12), (12, 14), (14, 16)]) # right lower


def remove_unused_landmarks(landmarks):
    landmarks = landmarks[10:]
    landmarks_start = landmarks[:7]
    landmarks_end = landmarks[13:]
    landmarks = landmarks_start + landmarks_end

    return landmarks


def retrieve_images(path):
    images_path = []

    for image in sorted(Path(f'{path}').glob('*.jpg'), key=os.path.getmtime):
        images_path.append(str(image))

    return images_path


def retrieve_frames(path):
    frames = []

    for image in sorted(Path(f'{path}').glob('*.jpg'), key=os.path.getmtime):
        frame = Image.open(image)
        frames.append(frame)

    return frames


def clear_csv_file(path):
    f = open(path, 'w')
    f.truncate()
    f.close()