import torch.nn as nn
import torch
import torch.nn.functional as F

def make_max_image(images, limit=0.5):
    # 입력이 3D 텐서라면 배치 차원을 추가
    if images.dim() == 3:
        images = images.unsqueeze(0)
        
    # 각 위치에서 최대값을 가진 채널의 인덱스 구하기
    _, max_indices = images.max(dim=1, keepdim=True)

    # 최대값이 XX 이상인 위치 찾기
    max_values = images.gather(1, max_indices)
    mask = max_values > limit

    # one-hot 인코딩을 사용하여 조건을 만족하는 위치에 대해 해당 채널을 1로 설정
    one_hot = torch.zeros_like(images)
    one_hot.scatter_(1, max_indices, mask.float())  # mask를 float으로 변환하여 사용
    
    # 채널중에 1의 개수가 N개 이하면 0으로 만들기
    for b in range(one_hot.size(0)):
        for c in range(one_hot.size(1)):
            if one_hot[b, c].sum() < 8:
                one_hot[b, c] = 0
    
    if one_hot.dim() == 4:
        one_hot.squeeze_(0)
        
    return one_hot

def to_gray(images):
    return torch.mean(images, dim=1, keepdim=True)

def classes_to_rgb(image):
    channels = image.shape[0]
    # print(f"Channels: {channels}")
    rgb_image = torch.zeros(3, image.size(1), image.size(2), dtype=torch.float32, device=image.device)
    color_diff = 1 / channels
    
    # image = make_max_image(image)
    
    for channel in range(channels):
        color = [0, 0, 0]
        
        if channel < channels / 2:
            color[0] = 1 - color_diff * channel
            color[1] = color_diff * channel
        else:
            color[1] = 1 - color_diff * channel
            color[2] = color_diff * channel

        if channel != 0:
            rgb_image[0] += image[channel] * color[0]
            rgb_image[1] += image[channel] * color[1]
            rgb_image[2] += image[channel] * color[2]
            
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    return rgb_image

def classes_to_rand_rgb(image):
    channels = image.shape[0]
    rgb_image = torch.zeros(3, image.size(1), image.size(2), dtype=torch.float32, device=image.device)
    
    # 미리 랜덤 색상을 생성합니다.
    random_colors = torch.rand(channels, 3, dtype=torch.float32, device=image.device)

    # 각 채널을 반복하지 않고 벡터화된 연산을 사용하여 rgb 이미지를 생성합니다.
    for i in range(3):  # RGB 채널을 위한 루프
        rgb_image[i] = torch.sum(image[:] * random_colors[:, i][:, None, None], dim=0)
    
    # 각 채널별로 평균 및 표준편차를 계산
    mean = rgb_image.view(3, -1).mean(dim=1)
    std = rgb_image.view(3, -1).std(dim=1)

    # 정규화: (값 - 평균) / 표준편차
    for i in range(3):
        rgb_image[i] = (rgb_image[i] - mean[i]) / std[i]

    # 정규화 후, 값의 범위를 0과 1 사이로 조정
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    return rgb_image


def classes_to_one_channel(image):
    channels = image.shape[0]
    gray_image = torch.zeros(1, image.size(1), image.size(2), dtype=torch.float32, device=image.device)
    
    for channel in range(channels):
        gray_image += image[channel]
        
    gray_image = (gray_image - gray_image.min()) / (gray_image.max() - gray_image.min())
    return gray_image


def calculate_image_sharpness(image):
    # Sobel 필터 정의
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device)

    # 이미지에 필터 적용
    edge_x = F.conv2d(image.unsqueeze(0), sobel_x, padding=1)
    edge_y = F.conv2d(image.unsqueeze(0), sobel_y, padding=1)

    # 엣지 강도 맵 생성
    edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)

    # # 선명도 계산 (예: 평균 엣지 강도)
    # sharpness = edge_magnitude.mean().item()
    
    # noise = edge_magnitude.std()
    # noise = noise.item()
    
    threshold = 0.5
    edge_only_image = torch.where(edge_magnitude > threshold, edge_magnitude, torch.tensor(0.0, device=edge_magnitude.device))
    edge_count = edge_only_image[edge_only_image > 0].numel()
    
    return edge_count

def calculate_nonzero_mean(image):
    # 이미지에서 0이 아닌 픽셀만 선택
    nonzero_pixels = image[image > 0]
    # 0이 아닌 픽셀의 평균값 계산
    nonzero_mean = nonzero_pixels.mean().item() if nonzero_pixels.numel() > 0 else 0
    return nonzero_mean