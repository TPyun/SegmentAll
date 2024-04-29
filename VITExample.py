# import torch
# import torchvision.transforms as transforms
# from torchvision.datasets import CIFAR10
# from torch.utils.data import DataLoader
# from timm import create_model
# import torch.nn as nn

# # 데이터셋을 불러오고 전처리하는 함수
# def get_dataloader(batch_size=64):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # ViT 입력 크기에 맞춰 이미지 크기 조정
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     return train_loader

# # Vision Transformer 모델 생성 및 학습
# def train_vit():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using {device} device")

#     # Timm 라이브러리에서 사전 훈련된 ViT 모델을 불러옴
#     model = create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
#     model = model.to(device)
#     model.train()

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

#     train_loader = get_dataloader()

#     # 간단한 학습 루프
#     for epoch in range(1):
#         for i, (images, labels) in enumerate(train_loader):
#             images = images.to(device)
#             labels = labels.to(device)

#             # Forward pass
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if (i+1) % 100 == 0:
#                 print(f'Epoch [{epoch+1}/{1}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# if __name__ == "__main__":
#     train_vit()

names = ['a', 'b', 'c', 'd', 'e']
for i in range(len(names)):
    names[i] = "dd"

print(names)