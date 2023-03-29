import cv2
import helpers

# filepaths
video_path = './videos/10second-test.mp4'
frame_path = "./images/frame_*.png"
gif_path = "./videos/image.gif"

vidcap = cv2.VideoCapture(video_path)


def video_to_frames(vidcap):
    success, image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(f'./images/frame_{count}.jpg', image)
      success, image = vidcap.read()
      count += 1


def image_to_gif(file_name):
    frames = helpers.retrieve_frames('./annotated_images/')

    frames[0].save(f'./videos/{file_name}.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=30)


#video_to_frames(vidcap)
image_to_gif('output')