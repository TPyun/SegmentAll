import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import functional as TF
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet
import timm
from torchvision.models import VisionTransformer
    
    
class CustomVisionTransformer(nn.Module):
    def __init__(self, num_classes=64, image_size=160):
        super(CustomVisionTransformer, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        
        self.vit = VisionTransformer(
            image_size=self.image_size,
            patch_size=16,
            num_layers=8,
            num_heads=8,
            hidden_dim=2048,
            mlp_dim=4096,
            num_classes=1  # This will be modified later
        )
        
        print(self.vit)
        self.transform_output = nn.Sequential(
            nn.Linear(self.vit.heads.head.in_features, self.num_classes * self.image_size//8 * self.image_size//8),
            nn.Unflatten(1, (self.num_classes, self.image_size//8, self.image_size//8)),
        )
        self.vit.heads = nn.Identity()
        
        self.decoder = nn.Sequential(
            nn.Upsample(size=(self.image_size, self.image_size), mode='bilinear', align_corners=False),
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.vit(x)
        # print(f"vit: {x.shape}")
        x = self.transform_output(x)
        # print(f"transform_output: {x.shape}")
        x = self.decoder(x)
        # print(f"decoder: {x.shape}")
        x = self.sigmoid(x)
        return x

class Translator(nn.Module):
    def __init__(self, channel):
        super(Translator, self).__init__()
        
        self.start = nn.Conv2d(3, 128, 9, 1, 4, padding_mode='reflect')
        self.start_bn = nn.InstanceNorm2d(128)
        self.start_relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(128, 256, 4, 2, 1, padding_mode='reflect')
        self.bn1 = nn.InstanceNorm2d(256)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(256, 512, 4, 2, 1, padding_mode='reflect')
        self.bn2 = nn.InstanceNorm2d(512)
        self.relu2 = nn.ReLU()

        self.mid_conv = nn.Conv2d(512, 512, 9, 1, 4, padding_mode='reflect')
        self.m_bn = nn.InstanceNorm2d(512)
        self.m_relu = nn.ReLU()
        
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn3 = nn.InstanceNorm2d(256)
        self.relu3 = nn.ReLU()
        
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn4 = nn.InstanceNorm2d(128)
        self.relu4 = nn.ReLU()
        
        self.end = nn.ConvTranspose2d(128, channel, 9, 1, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, log=False):
        x = self.start_relu(self.start_bn(self.start(x)))
        if log:
            print(f"start: {x.shape}")
            
        x = self.relu1(self.bn1(self.conv1(x)))
        if log:    
            print(f"conv2: {x.shape}")
            
        x = self.relu2(self.bn2(self.conv2(x)))
        if log:    
            print(f"conv3: {x.shape}")

        x = self.m_relu(self.m_bn(self.mid_conv(x)))
        if log:
            print(f"mid1_conv: {x.shape}")
            
        x = self.relu3(self.bn3(self.deconv1(x)))
        if log:
            print(f"deconv2: {x.shape}")
            
        x = self.relu4(self.bn4(self.deconv2(x)))
        if log:
            print(f"deconv3: {x.shape}")
            
        x = self.sigmoid(self.end(x))
        if log:
            print(f"end: {x.shape}")
        
        return x


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels, out_channels=out_channels, extra_blocks=None)
        print(self.fpn)
    def forward(self, x):
        return self.fpn(x)


class CustomDeepLabV3(nn.Module):
    def __init__(self, num_classes=1):
        super(CustomDeepLabV3, self).__init__()

        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        deeplabv3 = deeplabv3_resnet50(weights=weights)
        
        self.encoder = deeplabv3.backbone
        
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, 5, 1, 2)
        self.bn1 = nn.InstanceNorm2d(1024)
        self.relu1 = nn.ReLU()
        
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 5, 1, 2)
        self.bn2 = nn.InstanceNorm2d(512)
        self.relu2 = nn.ReLU()
        
        self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn3 = nn.InstanceNorm2d(256)
        self.relu3 = nn.ReLU()
        
        self.deconv4 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
        self.bn4 = nn.InstanceNorm2d(64)
        self.relu4 = nn.ReLU()
        
        self.end = nn.ConvTranspose2d(64, num_classes, 4, 2, 1)
        self.tanh = nn.Tanh()

        
    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x_conv = self.encoder.relu(x)
        x_maxpool = self.encoder.maxpool(x_conv)
        x_layer1 = self.encoder.layer1(x_maxpool)
        x_layer2 = self.encoder.layer2(x_layer1)
        x_layer3 = self.encoder.layer3(x_layer2)
        x_layer4 = self.encoder.layer4(x_layer3)
        
        # print(f'x_layer1: {x_layer1.shape}')
        # print(f'x_layer2: {x_layer2.shape}')
        # print(f'x_layer3: {x_layer3.shape}')
        # print(f'x_layer4: {x_layer4.shape}')
        
        '''
        x_layer1: torch.Size([1, 256, 256, 256])
        x_layer2: torch.Size([1, 512, 128, 128])
        x_layer3: torch.Size([1, 1024, 128, 128])
        x_layer4: torch.Size([1, 2048, 128, 128])
        '''
        
        x_deconv1 = self.relu1(self.bn1(self.deconv1(x_layer4))) + x_layer3
        x_deconv2 = self.relu2(self.bn2(self.deconv2(x_deconv1))) + x_layer2
        x_deconv3 = self.relu3(self.bn3(self.deconv3(x_deconv2))) + x_layer1
        x_deconv4 = self.relu4(self.bn4(self.deconv4(x_deconv3))) + x_conv
        x_tanh = self.tanh(self.end(x_deconv4)) / 2 + 0.5        
        return x_tanh



class ConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDownsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=8, stride=4, padding=2),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_masks=64, num_classes=4):
        super(SimpleSegmentationModel, self).__init__()
        self.num_masks = num_masks
        self.num_classes = num_classes
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        deeplabv3 = deeplabv3_resnet50(weights=weights)
        
        self.encoder = deeplabv3.backbone
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        
        in_channels = [256, 512, 2048]
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels, out_channels=num_masks)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(num_masks * 3, num_masks, kernel_size=1),
        )
        
        self.avgpool_64 = nn.AdaptiveAvgPool2d((64, 64))
        self.downsample = nn.Sequential(
            ConvDownsample(num_masks * 3, 512),
            ConvDownsample(512, 1024)
        )
        self.avgpool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_masks * num_classes)
        )
        
    def forward(self, x):
        origin_size = x.shape[2:]
        batch_size = x.shape[0]
        
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        
        x_1 = self.encoder.layer1(x)
        x_2 = self.encoder.layer2(x_1)
        x_3 = self.encoder.layer3(x_2)
        x_3 = self.encoder.layer4(x_3)
        
        fpn_out = self.fpn({'0': x_1, '1': x_2, '2': x_3})
        
        x_1_up = F.interpolate(fpn_out['0'], size=origin_size, mode="bilinear", align_corners=False)
        x_2_up = F.interpolate(fpn_out['1'], size=origin_size, mode="bilinear", align_corners=False)
        x_3_up = F.interpolate(fpn_out['2'], size=origin_size, mode="bilinear", align_corners=False)
        
        fpn_sum = torch.cat([x_1_up, x_2_up, x_3_up], dim=1)
        mask = self.decoder(fpn_sum)
        mask = torch.tanh(mask) / 2 + 0.5
        
        pooled = self.avgpool_64(fpn_sum)
        downsampled = self.downsample(pooled)
        pooled = self.avgpool_1(downsampled)
        class_logits = self.classifier_head(pooled)
        class_logits = class_logits.view(-1, self.num_masks, self.num_classes)
        class_probs = F.softmax(class_logits, dim=2)

        return mask, class_probs



class SingleChannelSeg(nn.Module):
    def __init__(self):
        super(SingleChannelSeg, self).__init__()
        
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        deeplabv3 = deeplabv3_resnet50(weights=weights)
        
        self.encoder = deeplabv3.backbone
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        
        in_channels = [256, 512, 2048]
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels, out_channels=1)
        
    def forward(self, x):
        origin_size = x.shape[2:]
        
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        
        x_1 = self.encoder.layer1(x)
        x_2 = self.encoder.layer2(x_1)
        x_3 = self.encoder.layer3(x_2)
        x_3 = self.encoder.layer4(x_3)
        
        x = self.fpn({'0': x_1, '1': x_2, '2': x_3})
        
        x['1'] = F.interpolate(x['1'], size=origin_size, mode="bilinear", align_corners=False)
        x['2'] = F.interpolate(x['2'], size=origin_size, mode="bilinear", align_corners=False)
        
        # plt.clf()
        # plt.figure(figsize=(10, 10))
        # plt.axis('off')
        # plt.imshow(x['0'][0, 0].detach().cpu().numpy())
        # plt.savefig("fpn_0.png")
        # plt.close()
        
        # plt.clf()
        # plt.figure(figsize=(10, 10))
        # plt.axis('off')
        # plt.imshow(x['1'][0, 0].detach().cpu().numpy())
        # plt.savefig("fpn_1.png")
        # plt.close()
        
        # plt.clf()
        # plt.figure(figsize=(10, 10))
        # plt.axis('off')
        # plt.imshow(x['2'][0, 0].detach().cpu().numpy())
        # plt.savefig("fpn_2.png")
        # plt.close()
        
        mask = x['0'] + x['1'] + x['2']
        mask = torch.tanh(mask) / 2 + 0.5
        return mask

    
class CustomSwinTransformer(nn.Module):
    def __init__(self, num_classes=64):
        super(CustomSwinTransformer, self).__init__()
        self.swin_transformer = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
        self.swin_transformer.patch_embed.proj = nn.Conv2d(64, self.swin_transformer.embed_dim, kernel_size=(4, 4), stride=(4, 4))
        
    def forward(self, x):
        x = self.swin_transformer(x)
        x = torch.sigmoid(x)
        return x
    
class HalfMade(nn.Module):
    def __init__(self, num_classes=64):
        super(HalfMade, self).__init__()
        weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        deeplabv3 = deeplabv3_resnet101(weights=weights)
        self.encoder = deeplabv3.backbone

        self.deconv1 = nn.ConvTranspose2d(2048, 1024, 4, 2, 1)
        self.bn1 = nn.InstanceNorm2d(1024)
        self.relu1 = nn.ReLU()
        
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.bn2 = nn.InstanceNorm2d(512)
        self.relu2 = nn.ReLU()
        
        self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn3 = nn.InstanceNorm2d(256)
        self.relu3 = nn.ReLU()
        
        self.end = nn.ConvTranspose2d(256, num_classes, 3, 1, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        encoded = self.encoder(x)['out']
        
        x = self.relu1(self.bn1(self.deconv1(encoded)))
        x = self.relu2(self.bn2(self.deconv2(x)))
        x = self.relu3(self.bn3(self.deconv3(x)))
        x = self.sig(self.end(x))
        return x
    
    
    
from torchvision.models import vit_b_16, ViT_B_16_Weights

class VisionTransformerSegmentation(nn.Module):
    def __init__(self, num_classes=64, img_size=224, hidden_dim=768):
        super(VisionTransformerSegmentation, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        # 사전 훈련된 ViT 모델 로드
        self.vit = vit_b_16(pretrained=True)
        
        # ViT의 출력 차원(hidden_dim)에 맞는 세그멘테이션 헤드 추가
        self.segmentation_head = nn.Sequential(
            nn.Linear(1000, 512),  # ViT의 출력 차원인 1000에서 시작
            nn.ReLU(),
            nn.Linear(512, 256 * (img_size // 4) * (img_size // 4))
        )
        
        self.deconv1 = nn.ConvTranspose2d(256, 512, 4, 2, 1)
        self.bn1 = nn.InstanceNorm2d(512)
        self.relu1 = nn.ReLU()
        
        self.deconv2 = nn.ConvTranspose2d(512, 1024, 4, 2, 1)
        self.bn2 = nn.InstanceNorm2d(1024)
        self.relu2 = nn.ReLU()
        
        self.deconv3 = nn.ConvTranspose2d(1024, 256, 4, 2, 1)
        self.bn3 = nn.InstanceNorm2d(256)
        self.relu3 = nn.ReLU()
        
        self.end = nn.ConvTranspose2d(256, num_classes, 3, 1, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        # ViT 모델의 마지막 층 출력
        x = self.vit(x)
        # 4, 1000
        
        # 세그멘테이션 헤드를 통해 예측
        x = self.segmentation_head(x)
        
        # 세그멘테이션 맵 크기로 재조정
        x = x.view(-1, self.num_classes, self.img_size // 4, self.img_size // 4)
        
        x = self.relu1(self.bn1(self.deconv1(x)))
        x = self.relu2(self.bn2(self.deconv2(x)))
        x = self.relu3(self.bn3(self.deconv3(x)))
        x = self.sig(self.end(x))
        
        return x