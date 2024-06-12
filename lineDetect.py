from math import dist
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time

def detect_lines(image):
    # 2진 이미지를 불러옵니다.
    # image_path = '/home/wooyung/Develop/RadarDetection/WallLineSeg/Eval/instance.png'  # 여기에 2진 이미지 파일 경로를 입력합니다.
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # resize
    # image = cv2.resize(image, (512, 512))
    original_image = image.copy()

    start_time = time.time()
    
    
    # 이미지에서 상담 절반은 0으로 채웁니다.
    image[:image.shape[1] // 3, :] = 0
    # 하단의 1/3은 0으로 채웁니다.
    image[image.shape[1] * 65 // 100:, :] = 0
    
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # y축을 기준으로 맨 아래에 있는 1값만 남기고 그 위는 다 0으로 바꾼다. for문 거꾸로 돌리기
    for x in range(binary_image.shape[1]):
        for y in range(binary_image.shape[0] - 1, 0, -1):
            if binary_image[y, x]!= 0:
                binary_image[:y, x] = 0
                break

    kernel = np.ones((2, 2), np.uint8)  # 3x3 크기의 사각형 커널
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)

    # 허프 변환을 사용하여 직선을 탐지합니다.
    lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, threshold=0, minLineLength=8, maxLineGap=4)

    if lines is None:
        print('No lines detected')
        return
    
    horizenal_lines = []
    for i, line in enumerate(lines):
        # 각도 60 수평에 가까운 것만 남기기
        angle = np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0]) * 180 / np.pi
        angle = np.abs(angle)
        if 60 < angle < 120:
            continue
        horizenal_lines.append(line)
        
    # # 선의 길이가 긴 순으로 정렬
    # horizenal_lines = sorted(horizenal_lines, key=lambda x: np.sqrt((x[0][2] - x[0][0]) ** 2 + (x[0][3] - x[0][1]) ** 2), reverse=True)
        
    # # 각각의 각도 차이를 계산하여 각도차이가 작은것에 대해 nms를 적용합니다.
    # nms_lines = []
    # for i, line in enumerate(horizenal_lines):
    #     x1, y1, x2, y2 = line[0]
    #     if len(nms_lines) == 0:
    #         nms_lines.append(line)
    #     else:
    #         is_new_line = True
    #         for nms_line in nms_lines:
    #             nx1, ny1, nx2, ny2 = nms_line[0]
    #             angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    #             n_angle = np.arctan2(ny2 - ny1, nx2 - nx1) * 180 / np.pi
                
    #             if np.abs(angle - n_angle) < 20:
    #                 distance = np.sqrt((nx1 - x1) ** 2 + (ny1 - y1) ** 2) + np.sqrt((nx2 - x2) ** 2 + (ny2 - y2) ** 2)
    #                 if distance < 100:
    #                     is_new_line = False
    #                     break
                    
    #         if is_new_line:
    #             nms_lines.append(line)
    
    # end_time = time.time()
    # print('Elapsed time:', end_time - start_time)
                
    # print('Detected lines:', len(nms_lines))

    line_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    for line in horizenal_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
    # for line in nms_lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    # plt.figure(figsize=(12, 12))
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
    # plt.savefig('/home/wooyung/Develop/RadarDetection/WallLineSeg/lineDetect.png', bbox_inches='tight', pad_inches=0)
    # plt.close()
    
    return line_image
    

if __name__ == '__main__':
    detect_lines('/home/wooyung/Develop/RadarDetection/WallLineSeg/Eval/instance.png')