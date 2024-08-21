import cv2
import os

directory = '/home/wooyung/Develop/RadarDetection/SegmentAll/Eval not sorted'
images = sorted([img for img in os.listdir(directory) if img.endswith(".png")])
print(len(images))

# 첫 번째 이미지를 읽어서 비디오의 프레임 크기를 설정
frame = cv2.imread(os.path.join(directory, images[0]))
height, width, layers = frame.shape

# 비디오 파일 이름과 경로 설정
video_name = '/home/wooyung/Develop/RadarDetection/SegmentAll/not sorted.mp4'

# fourcc 코드 설정 부분 수정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 올바른 방법으로 fourcc 코드를 얻습니다.
video = cv2.VideoWriter(video_name, fourcc, 2, (width, height))

for image in images:
    frame = cv2.imread(os.path.join(directory, image))
    if frame is not None:
        print(f"Writing frame {image}")
        video.write(frame)
    else:
        print(f"Failed to load image: {image}")
video.release()
print("Video file has been successfully created.")

# 비디오 열기
cap = cv2.VideoCapture(video_name)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# import cv2
# import numpy as np

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video = cv2.VideoWriter('/home/wooyung/Develop/RadarDetection/SegmentAll/sorted.mp4', fourcc, 20.0, (640, 480))

# if video.isOpened():
#     for i in range(100):
#         frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
#         video.write(frame)
#     video.release()
#     print("Video created successfully.")
# else:
#     print("Failed to create video. Check codec and file path.")