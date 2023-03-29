from test import generate_landmarks_annotations
from helpers import PoseLandmark, POSE_CONNECTIONS
import numpy as np

# TODO: retrieve images
test_im1 = './test_images/test-1.jpg'
test_im2 = './test_images/test-2.jpeg'
test_im3 = './test_images/test-2-same.jpeg'
test_im4 = './test_images/test-3.jpeg'
# TODO: retrieve landmarks
user_landmarks = generate_landmarks_annotations('user', [test_im2])
guide_landmarks = generate_landmarks_annotations('guide', [test_im1])

percentage_dif_arr = []
cos_sim_arr = []
# calculate gradient on limb connections
for pose_pair in POSE_CONNECTIONS:
    print(f'Calculating pose pair {pose_pair}: ')
    # get key for landmark dictionary from pose pair
    first_idx = pose_pair[0]
    second_idx = pose_pair[1]
    first_key = list(user_landmarks)[first_idx]
    second_key = list(user_landmarks)[second_idx]
    print(f'user pose {first_idx} {user_landmarks[first_key]}')
    print(f'user pose {second_idx} {user_landmarks[second_key]}')
    # get x and y coordinates from user key poses
    user_x1 = user_landmarks[first_key][0]
    user_y1 = user_landmarks[first_key][1]
    user_x2 = user_landmarks[second_key][0]
    user_y2 = user_landmarks[second_key][1]
    '''print(f'user x1 coordinate: {user_x1}')
    print(f'user y1 coordinate: {user_y1}')
    print(f'user x2 coordinate: {user_x2}')
    print(f'user y2 coordinate: {user_y2}')'''

    print(f'guide pose {first_idx} {guide_landmarks[first_key]}')
    print(f'guide pose {second_idx} {guide_landmarks[second_key]}')
    # get x and y coordinates from second key pose
    guide_x1 = guide_landmarks[first_key][0]
    guide_y1 = guide_landmarks[first_key][1]
    guide_x2 = guide_landmarks[second_key][0]
    guide_y2 = guide_landmarks[second_key][1]
    '''print(f'guide x1 coordinate: {guide_x1}')
    print(f'guide y1 coordinate: {guide_y1}')
    print(f'guide x2 coordinate: {guide_x2}')
    print(f'guide y2 coordinate: {guide_y2}')'''

    user_gradient = (user_y2 - user_y1) / (user_x2 - user_x1)
    guide_gradient = (guide_y2 - guide_y1) / (guide_x2 - guide_x1)

    print(f'user gradient: {user_gradient}')
    print(f'guide gradient: {guide_gradient}')

    # calculate percentage difference on corresponding image
    percentage_dif = ((abs(guide_gradient - user_gradient)) / ((guide_gradient + user_gradient) / 2))

    percentage_dif_arr.append(percentage_dif)
    print(f'Difference: {percentage_dif}')


print(f'Similarity: {100 - (np.mean(percentage_dif_arr) * 100)}')


# TODO: save calculations to csv file

# TODO: cosine similarity
def cosine_similarity(user_coord1, user_coord2, guide_coord1, guide_coord2):
    '''user_limb = (user_x2 - user_x1, user_y2 - user_y1)
        guide_limb = (guide_x2 - guide_x1, guide_y2 - guide_y1)
        user_dist = np.sqrt((user_x2 - user_x1)**2 + (user_y2 - user_y1)**2)
        guide_dist = np.sqrt((guide_x2 - guide_x1)**2 + (guide_y2 - guide_y1)**2)
        # normalise gradient by length of limb
        user_grad_norm = user_gradient / user_dist
        guide_grad_norm = guide_gradient / guide_dist
        #user_limb_norm = user_limb / user_dist
        #guide_limb_norm = guide_limb / guide_dist
        cos_sim = np.dot(user_grad_norm, guide_grad_norm) / (np.linalg.norm(guide_grad_norm) * np.linalg.norm(user_grad_norm))
        #cos_sim = np.dot(user_limb_norm, guide_limb_norm) / (np.linalg.norm(guide_limb_norm) * np.linalg.norm(user_limb_norm))
        cos_sim_arr.append(cos_sim)
        print(f'cosine similarity: {cos_sim}')
        print(cos_sim_arr)'''
