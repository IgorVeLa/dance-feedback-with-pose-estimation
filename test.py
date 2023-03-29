import cv2
import math
import mediapipe as mp
import helpers
import csv
import pandas as pd

mp_pose = mp.solutions.pose
mp.solutions.pose.PoseLandmark = helpers.PoseLandmark
mp.solutions.pose.POSE_CONNECTIONS = helpers.POSE_CONNECTIONS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

DESIRED_WIDTH = 480
DESIRED_HEIGHT = 480


def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))

    # cv2.imshow('', img)
    cv2.imwrite(f'./annotated_images/.jpg', img)
    cv2.waitKey(0)


# Read images with OpenCV.
# image_files = helpers.retrieve_images('./images/')
#image_files = ['./test_images/test-2.jpeg']
image_files = ['./test_images/test-2-same.jpeg']


def generate_landmarks_annotations(video_type, image_files):
    landmark_data = {
        'NOSE': (0, 0),
        'LEFT_SHOULDER': None,
        'RIGHT_SHOULDER': None,
        'LEFT_ELBOW': None,
        'RIGHT_ELBOW': None,
        'LEFT_WRIST': None,
        'RIGHT_WRIST': None,
        'LEFT_HIP': None,
        'RIGHT_HIP': None,
        'LEFT_KNEE': None,
        'RIGHT_KNEE': None,
        'LEFT_ANKLE': None,
        'RIGHT_ANKLE': None,
        'LEFT_HEEL': None,
        'RIGHT_HEEL': None,
        'LEFT_FOOT_INDEX': None,
        'RIGHT_FOOT_INDEX': None
    }

    if video_type == 'user':
        data_path = './data/user_landmarks.csv'
        helpers.clear_csv_file(data_path)
    elif video_type == 'guide':
        data_path = './data/guide_landmarks.csv'
        helpers.clear_csv_file(data_path)
    else:
        return 'Please specify which video type you\'re inputting'

    with mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=.5,
            model_complexity=2) as pose:
        for idx, image in enumerate(image_files):
            image = cv2.imread(image)
            # Convert the BGR image to RGB and process it with MediaPipe Pose.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            image_height, image_width, _ = image.shape

            # replace MP 33 point to custom 17 point
            landmarks = results.pose_landmarks.landmark
            landmarks = helpers.remove_unused_landmarks(landmarks)
            del results.pose_landmarks.landmark[:]
            results.pose_landmarks.landmark.extend(landmarks)

            landmark_data['LEFT_SHOULDER'] = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width,
                                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height)
            landmark_data['RIGHT_SHOULDER'] = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image_width,
                                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * image_height)
            landmark_data['LEFT_ELBOW'] = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image_width,
                                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image_height)
            landmark_data['RIGHT_ELBOW'] = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * image_width,
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * image_height)
            landmark_data['LEFT_WRIST'] = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width,
                                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height)
            landmark_data['RIGHT_WRIST'] = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width,
                                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height)
            landmark_data['LEFT_HIP'] = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width,
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height)
            landmark_data['RIGHT_HIP'] = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * image_width,
                                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * image_height)
            landmark_data['LEFT_KNEE'] = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width,
                                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height)
            landmark_data['RIGHT_KNEE'] = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width,
                                           landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_height)
            landmark_data['LEFT_ANKLE'] = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image_width,
                                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image_height)
            landmark_data['RIGHT_ANKLE'] = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * image_width,
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * image_height)
            landmark_data['LEFT_HEEL'] = (landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x * image_width,
                                          landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y * image_height)
            landmark_data['RIGHT_HEEL'] = (landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x * image_width,
                                           landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y * image_height)
            landmark_data['LEFT_FOOT_INDEX'] = (landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x * image_width,
                                                landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * image_height)
            landmark_data['RIGHT_FOOT_INDEX'] = (landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x * image_width,
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * image_height)

            # Draw pose landmarks.
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=10, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 100, 100), thickness=10, circle_radius=2)
            )

            # save images with annotations
            # cv2.imwrite(f'./annotated_images/image_{idx}.jpg', annotated_image)
            cv2.imwrite(f'./test_images/{str(video_type)}_test_annotated_{idx}.jpg', annotated_image)
            # print(f'image_{idx} saved')

            # Save landmarks
            with open(data_path, 'a', newline='') as csvfile:
                fieldnames = pd.DataFrame(columns=['frame_idx', 'keypoint', 'x', 'y'])

                # add field names to csv file as columns
                fieldnames.to_csv(csvfile, index=False)

                # add frame information to csv file
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for key in landmark_data:
                    writer.writerow({'frame_idx': idx,
                                     'keypoint': key,
                                     'x': landmark_data[key][0],
                                     'y': landmark_data[key][1]})

            return landmark_data

#generate_landmarks_annotations('guide')