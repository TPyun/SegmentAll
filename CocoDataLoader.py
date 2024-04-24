import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import random
from torchvision import transforms
import json
import cv2
from pycocotools import mask as maskUtils
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
from torchvision.transforms.functional import resize
import os.path as osp
from PIL import ImageFilter

class SegDataset(torch.utils.data.Dataset):
    def __init__(self, path, width, height, mode='train'):
        raw_set = []
        
        self.width = width
        self.height = height

        self.mode = mode
        self.image_set = []
        
        self.max_masks_dict = self.get_max_masks_dict(path)
        print(f"Max masks dict: { self.max_masks_dict}")    
            
        self.class_list = []
        for key, value in self.max_masks_dict.items():
            for i in range(value):
                self.class_list.append(key)
                
        self.object_num = len(self.class_list) + 1
        self.class_num = 1 + len(self.max_masks_dict)
        
        if self.mode == 'train' and os.path.exists('/home/wooyung/Develop/RadarDetection/train_set.pt'):
            self.image_set = torch.load('/home/wooyung/Develop/RadarDetection/train_set.pt')
            print(f"Train set loaded: {len(self.image_set)}")
            return
        elif self.mode == 'eval' and os.path.exists('/home/wooyung/Develop/RadarDetection/eval_set.pt'):
            self.image_set = torch.load('/home/wooyung/Develop/RadarDetection/eval_set.pt')
            print(f"Eval set loaded: {len(self.image_set)}")
            return

        for folder in os.listdir(path):
            if not os.path.isdir(os.path.join(path, folder)):
                continue
            
            # 각각의 case 열기
            for sub_folder in os.listdir(os.path.join(path, folder)):
                # 여기에 task 0, 1, 2, json 있음
                # task 0 front camera만 할거임
                if 'task_0' not in sub_folder:#  and 'task_1' not in sub_folder:
                    continue
                
                for file in os.listdir(os.path.join(path, folder, sub_folder)):
                    ann_json_name = 'annotations.json'
                    with open(osp.join(path, folder, sub_folder, ann_json_name), 'r') as ann_json_file:
                        ann_json_data = json.load(ann_json_file)
                    
                    if 'data' in file:
                        # 여기서 이미지 목록이랑 사이즈 가져옴
                        image_list_in_file_list = []
                        mani_json_name = 'manifest.jsonl'  
                        
                        if not os.path.exists(osp.join(path, folder, sub_folder, file, mani_json_name)):
                            continue
                        
                        with open(osp.join(path, folder, sub_folder, file, mani_json_name), 'r') as mani_json_file:
                            for line in mani_json_file:
                                mani_json_line = json.loads(line)
                                try:
                                    image_list_in_file_list.append([mani_json_line['name'], mani_json_line['width'], mani_json_line['height']])
                                except Exception as e:
                                    continue
                        
                        # json에서 가져온 이미지 목록으로 이미지 가져옴
                        for image_i, image_info in enumerate(image_list_in_file_list):
                            image =Image.open(osp.join(path, folder, sub_folder, file, image_info[0] + '.jpg'))
                            image = image.filter(ImageFilter.DETAIL)
                            image = np.array(image)
                            
                            image = torch.from_numpy(image).permute(2, 0, 1).float()
                            image /= 255.0
                            semantic_image = self.get_seg_image_from_annotation(ann_json_data, image_i, image_info[1], image_info[2], 'SE')
                            instance_image = self.get_seg_image_from_annotation(ann_json_data, image_i, image_info[1], image_info[2], 'OD')
                            
                            # 이미지 128 128로 리사이즈
                            image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
                            semantic_image = torch.nn.functional.interpolate(semantic_image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
                            instance_image = torch.nn.functional.interpolate(instance_image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
                            raw_set.append([image, semantic_image, instance_image])
                            # print(f"Image: {image.shape}, Semantic: {semantic_image.shape}, Instance: {instance_image.shape}")

        if self.mode == 'train':
            self.image_set = raw_set[:int(len(raw_set) * 0.8)]
            torch.save(self.image_set, '/home/wooyung/Develop/RadarDetection/train_set.pt')
        elif self.mode == 'eval':
            self.image_set = raw_set[int(len(raw_set) * 0.8):]
            torch.save(self.image_set, '/home/wooyung/Develop/RadarDetection/eval_set.pt')
                
        print(f"Number of images: {len(self.image_set)}")

        
    def random_effect(self, index):
            
        image = self.image_set[index][0]
        semantic_image = self.image_set[index][1]
        instance_image = self.image_set[index][2]
        
        scale = random.uniform(1.0, 1.5)
        width = int(self.width * scale)
        height = int(self.width * scale)
        
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
        semantic_image = torch.nn.functional.interpolate(semantic_image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
        instance_image = torch.nn.functional.interpolate(instance_image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
        
        x = random.randint(0, width - self.width)
        y = random.randint(0, height - self.height)
        
        image = image[:, y:y+self.height, x:x+self.width]
        semantic_image = semantic_image[:, y:y+self.height, x:x+self.width]
        instance_image = instance_image[:, y:y+self.height, x:x+self.width]
        
        if random.randint(0, 1) == 0:
            return [image, semantic_image, instance_image]
        
        image = torch.flip(image, [2])
        semantic_image = torch.flip(semantic_image, [2])
        instance_image = torch.flip(instance_image, [2])
        
        return [image, semantic_image, instance_image]

                                
    def __len__(self):
        return len(self.image_set) // 4 * 4

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.random_effect(index)
        elif self.mode == 'eval':
            return self.image_set[index]

    def get_class_num(self):
        return self.class_num
    
    def get_object_num(self):
        return self.object_num
    
    def get_class_channel_list(self):
        channel_list = ['background']
        for key in self.max_masks_dict.keys():
            channel_list.append(key)
        return channel_list
    
    def get_max_masks_dict(self, path):
        max_masks_dict = {}
        for folder in os.listdir(path):
            if not os.path.isdir(os.path.join(path, folder)):
                continue
            
            # 각각의 case 열기
            for sub_folder in os.listdir(os.path.join(path, folder)):
                # 여기에 task 0, 1, 2, json 있음
                # task 0 front camera만 할거임
                if 'task_0' not in sub_folder:#  and 'task_1' not in sub_folder:
                    continue
                
                for file in os.listdir(os.path.join(path, folder, sub_folder)):
                    ann_json_name = 'annotations.json'
                    with open(osp.join(path, folder, sub_folder, ann_json_name), 'r') as ann_json_file:
                        ann_json_data = json.load(ann_json_file)
                    
                    if 'data' in file:
                        # 여기서 이미지 목록이랑 사이즈 가져옴
                        image_list_in_file_list = []
                        mani_json_name = 'manifest.jsonl'
                        
                        if not os.path.exists(osp.join(path, folder, sub_folder, file, mani_json_name)):
                            continue
                        
                        with open(osp.join(path, folder, sub_folder, file, mani_json_name), 'r') as mani_json_file:
                            for line in mani_json_file:
                                mani_json_line = json.loads(line)
                                try:
                                    image_list_in_file_list.append([mani_json_line['name'], mani_json_line['width'], mani_json_line['height']])
                                except Exception as e:
                                    continue
                        
                        # json에서 가져온 이미지 목록으로 이미지 가져옴
                        for image_i, image_info in enumerate(image_list_in_file_list):
                            
                            masks_dict = {}
                            # json에서 이미지 번호로 찾기 (한 이미지에 여러 클래스 있음)
                            for main_data in ann_json_data:
                                for sub_data in main_data['shapes']:
                                    if sub_data['frame'] != image_i:
                                        continue
                                    
                                    if sub_data['label'] not in masks_dict:
                                        masks_dict[sub_data['label']] = 1
                                    else:
                                        masks_dict[sub_data['label']] += 1

                            # max_masks_dict에서 masks_dict의 최댓값을 업데이트
                            for key in masks_dict:
                                if key not in max_masks_dict:
                                    max_masks_dict[key] = masks_dict[key]
                                else:
                                    max_masks_dict[key] = max(max_masks_dict[key], masks_dict[key])
                
        return max_masks_dict

            
    def get_label_channel(self, label):
        for i, key in enumerate(self.max_masks_dict.keys()):
            if key == label:
                return i + 1

        
    def get_seg_image_from_annotation(self, json_data, frame_num, width, height, mode):
        if mode == 'SE':
            seg_image = np.zeros((height, width, self.class_num))
        elif mode == 'OD':
            seg_image = np.zeros((height, width, self.object_num))
        else:
            print("Invalid mode!")
            return None
        
        background = np.ones((height, width))
        mask_iter = 1
        # print("====================================")
        # json에서 이미지 번호로 찾기 (한 이미지에 여러 클래스 있음)
        for main_data in json_data:
            for sub_data in main_data['shapes']:
                if sub_data['frame'] != frame_num:
                    continue
                
                # 포인트 xy변환하고 세그 이미지에 넣기
                points = sub_data['points']
                processed_point = []
                for x, y in zip(points[::2], points[1::2]):
                    processed_point.append([x, y])
                    
                processed_point = np.array(processed_point, np.int32)
                
                # 해당 라벨에 대한 채널만을 위한 마스크 생성
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [processed_point], (1))
                
                # 마스크를 사용하여 적절한 채널에만 색상 적용
                if mode == 'SE':
                    label = sub_data['label']
                    label_number = self.get_label_channel(label)
                    
                elif mode == 'OD':
                    # print(f"Object: {sub_data['label']}, mask_iter: {mask_iter}")
                    label_number = mask_iter
                    mask_iter += 1
                
                seg_image[:, :, label_number][mask == 1] = 1
                
                # 백그라운드에서 mask 지우기
                background[mask == 1] = 0
                
                    
        # 백그라운드 채널에 넣기
        seg_image[:, :, 0] = background
        
        # 채널 램덤으로 섞끼
        if mode == 'OD':
            seg_image = seg_image.transpose(2, 0, 1)
            np.random.shuffle(seg_image)
            seg_image = seg_image.transpose(1, 2, 0)
                
        # plt.clf()
        # plt.imshow(seg_image)
        # plt.savefig('/home/wooyung/Develop/RadarDetection/seg.png')
        seg_image = torch.from_numpy(seg_image).permute(2, 0, 1).float()
        # seg_image /= 255.0
        return seg_image


class InstanceCocoDataset(Dataset):
    def __init__(self, root, dataType, width, height):
        
        self.root = root
        self.dataType = dataType
        annotation = '{}/instance_annotations/instances_new_{}.json'.format(self.root, dataType)
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        
        self.width = width
        self.height = height
        
        print(f"{dataType} Number of images: {len(self.ids)}")
        
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        img_file = coco.loadImgs(img_id)[0]['file_name']
        img = io.imread(os.path.join(self.root, self.dataType, img_file))
        img = torch.tensor(img, dtype=torch.float32)
        if len(img.shape) == 2:
            img = img.unsqueeze(2).expand(-1, -1, 3)
        img = img.permute(2, 0, 1)
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(self.width, self.height), mode='bilinear', align_corners=True).squeeze(0)
        img = img / 255.0
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        # 타겟 데이터 구성
        instance_seg = torch.zeros((64, self.width, self.height), dtype=torch.float32)
        
        for i, ann in enumerate(coco_annotation):
            mask = coco.annToMask(ann)
            category = ann['category_id']
            mask = torch.tensor(mask, dtype=torch.float32) 
            mask = mask.unsqueeze(0)
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(self.width, self.height), mode='bilinear', align_corners=True).squeeze(0)
            # 마스크 부분 1로 채우기
            if i > 63:
                print(f"Over 63: {i}")
                continue
            instance_seg[i] = torch.where(mask == 1, 1, instance_seg[i])

        if 'train' in self.dataType:
            img, instance_seg = self.random_effect(img, instance_seg)
        
        return img, instance_seg

    def __len__(self):
        # max = 512
        # if len(self.ids) > max:
        #     return max
        return len(self.ids)

    def random_effect(self, image, instance_image):
        scale = random.uniform(1.0, 1.5)
        width = int(self.width * scale)
        height = int(self.width * scale)

        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
        instance_image = torch.nn.functional.interpolate(instance_image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
        x = random.randint(0, width - self.width)
        y = random.randint(0, height - self.height)
        image = image[:, x:x+self.width, y:y+self.height]
        instance_image = instance_image[:, x:x+self.width, y:y+self.height]
        
        if random.randint(0, 1) == 0:
            return [image, instance_image]
        image = torch.flip(image, [2])
        instance_image = torch.flip(instance_image, [2])
        return image, instance_image
    
    
class PanopticCocoDataset(Dataset):
    def __init__(self, root, dataType, width, height):
        self.root = root
        self.dataType = dataType
        self.image_folder = '{}/{}'.format(self.root, dataType)
        self.annotation_file = '{}/panoptic_annotations/panoptic_{}.json'.format(self.root, dataType)
        self.panoptic_image_folder = '{}/panoptic_annotations/panoptic_{}/'.format(self.root, dataType)
        self.image_list = os.listdir(self.image_folder)
        self.image_list.sort()
        
        self.width = width
        self.height = height
        
        # JSON 파일 로드
        with open(self.annotation_file, 'r') as f:
            self.panoptic_data = json.load(f)

        # 총 개수 확인
        print(f"Number of annotations: {len(self.panoptic_data['annotations'])}")

        if len(os.listdir(self.image_folder)) == len(self.panoptic_data['annotations']):
            self.length = len(self.panoptic_data['annotations'])
            print(f"Number of images: {self.length}")
        else:
            print("ERROR: Number of images and annotations are not matched!!!")
        
        
    def __getitem__(self, index):
        # 폴더 안에 이미지 index번째 이미지 정보를 가져옴
        image = io.imread(os.path.join(self.image_folder, self.image_list[index]))
        image = torch.tensor(image, dtype=torch.float32)
        image = image / 255.0
        if len(image.shape) == 2:
            image = image.unsqueeze(2).expand(-1, -1, 3)
        image = image.permute(2, 0, 1)
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(self.width, self.height), mode='bilinear', align_corners=True).squeeze(0)
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])(image)
        
        
        # 예제로 첫 번째 이미지 정보를 사용
        image_info = self.panoptic_data['annotations'][index]
        image_id = image_info['image_id']
        file_name = image_info['file_name']

        # Panoptic 이미지 로드
        panoptic_image_path = self.panoptic_image_folder + file_name
        panoptic_img = cv2.imread(panoptic_image_path, cv2.IMREAD_COLOR)
        panoptic_img = cv2.cvtColor(panoptic_img, cv2.COLOR_BGR2RGB)

        # 고유 ID를 계산하기 위해 R, G, B 채널 사용
        panoptic_id = panoptic_img[:, :, 0].astype(np.uint32) + \
                    panoptic_img[:, :, 1].astype(np.uint32) * 256 + \
                    panoptic_img[:, :, 2].astype(np.uint32) * 256 * 256
        
        if len(image_info['segments_info']) > 63:
            instance_seg = torch.zeros((len(image_info['segments_info']), self.width, self.height), dtype=torch.float32)
        else:
            instance_seg = torch.zeros((64, self.width, self.height), dtype=torch.float32)

        # 고유 ID를 사용하여 객체별 세그멘테이션 정보 추출
        for i, segment_info in enumerate(image_info['segments_info']):
            segment_id = segment_info['id']
            category_id = segment_info['category_id']
            
            segment_mask = panoptic_id == segment_id
            segment_mask = segment_mask.astype(np.float32)

            mask = torch.tensor(segment_mask, dtype=torch.float32)
            mask = mask.unsqueeze(0)
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(self.width, self.height), mode='bilinear', align_corners=True).squeeze(0)
            
            instance_seg[i] = torch.where(mask == 1, 1, instance_seg[i])
            
        instance_seg = instance_seg[instance_seg.sum(dim=(1, 2)).argsort(descending=True)]
        instance_seg = instance_seg[:64]
            
        # if 'train' in self.dataType:
        #     image, instance_seg = self.random_effect(image, instance_seg)
    
        return image, instance_seg
    
    def __len__(self):
        # if 'train' in self.dataType:
        #     return 16000
        # elif 'val' in self.dataType:
        #     return 1024
            
        return self.length // 16 * 16

    
    def random_effect(self, image, instance_image):
        scale = random.uniform(1.0, 1.2)
        width = int(self.width * scale)
        height = int(self.width * scale)

        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
        instance_image = torch.nn.functional.interpolate(instance_image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
        x = random.randint(0, width - self.width)
        y = random.randint(0, height - self.height)
        image = image[:, x:x+self.width, y:y+self.height]
        instance_image = instance_image[:, x:x+self.width, y:y+self.height]
        
        if random.randint(0, 1) == 0:
            return [image, instance_image]
        image = torch.flip(image, [2])
        instance_image = torch.flip(instance_image, [2])
        return image, instance_image


class SAMDataset(Dataset):
    def __init__(self, root, type, width, height):
        self.root = root
        self.width = width
        self.height = height
        self.type = type
        # root 폴더 안에 있는 모든 json 파일을 가져옴
        self.json_files = [f for f in os.listdir(self.root) if f.endswith('.json')]
        self.json_files.sort()
        
        print(f"Total Number of annotations: {len(self.json_files)}")
        
        # train val 비율 0.1
        if type == 'train':
            self.json_files = self.json_files[:int(len(self.json_files) * 0.9)]
        elif type == 'val':
            self.json_files = self.json_files[int(len(self.json_files) * 0.9):]
        
        # 총 개수 확인
        print(f"{type} Number of annotations: {len(self.json_files)}")
        
        self.transform = Compose([
            ToTensor(),
            Resize((self.width, self.height), antialias=None),
        ])
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __len__(self):
        return len(self.json_files) // 4 * 4
        
    def __getitem__(self, index):
        # json 파일을 하나씩 가져옴
        with open(os.path.join(self.root, self.json_files[index]), 'r') as f:
            data = json.load(f)
        
        image_path = os.path.join(self.root, data['image']['file_name'])
        image = Image.open(image_path)  # PIL.Image.open을 사용하여 이미지를 불러옴

        # 만약 이미지가 그레이스케일이면 RGB로 변환
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 정의된 변환 적용
        image = self.transform(image)
        
        instance_seg = self.create_segmentation_tensor(data)

        if 'train' in self.type:
            image, instance_seg = self.random_effect(image, instance_seg)
        
        image_mean_brightness = image.mean()
        image = image - image_mean_brightness + 0.5
        image = self.normalize(image)
        
        return image, instance_seg

    def create_segmentation_tensor(self, data):
        annotation = data['annotations']
        if len(annotation) > 64:
            segmentation_tensor = torch.zeros(len(annotation), self.width, self.height, dtype=torch.float32)
        else:
            segmentation_tensor = torch.zeros(64, self.width, self.height, dtype=torch.float32)
            
        for i, ann in enumerate(annotation):
            segmentation = ann['segmentation']
            encoded_mask = {'size': segmentation['size'], 'counts': segmentation['counts']}
            mask = maskUtils.decode(encoded_mask)  # np.array(dtype=uint8) mask
            mask = torch.tensor(mask, dtype=torch.float32)
            mask.permute(1, 0)
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(self.width, self.height), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            mask = torch.where(mask > 0.5, 1, 0)

            segmentation_tensor[i] = mask
            
        # mask의 크기를 기준으로 정렬
        segmentation_tensor = segmentation_tensor[segmentation_tensor.sum(dim=(1, 2)).argsort(descending=True)]
                
        return segmentation_tensor[:64]
    
    def random_effect(self, image, instance_image):
        color_jitter = transforms.ColorJitter(brightness=(0.5, 1.5),contrast=(0.5, 1.5),saturation=(0.5, 1.5))
        image = color_jitter(image)
        
        scale = random.uniform(1.0, 1.2)
        width = int(self.width * scale)
        height = int(self.height * scale)

        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(width, height), mode='nearest').squeeze(0)
        instance_image = torch.nn.functional.interpolate(instance_image.unsqueeze(0), size=(width, height), mode='nearest').squeeze(0)
        x = random.randint(0, width - self.width)
        y = random.randint(0, height - self.height)
        image = image[:, x:x+self.width, y:y+self.height]
        instance_image = instance_image[:, x:x+self.width, y:y+self.height]
        
        
        flip_seed = random.randint(0, 3)
        if flip_seed == 0:
            return [image, instance_image]
        elif flip_seed == 1:
            image = torch.flip(image, [2])
            instance_image = torch.flip(instance_image, [2])
        elif flip_seed == 2:
            image = torch.flip(image, [1])
            instance_image = torch.flip(instance_image, [1])
        else:
            image = torch.flip(image, [1, 2])
            instance_image = torch.flip(instance_image, [1, 2])
        
        return image, instance_image
    
    
class MapillaryData(Dataset):
    def __init__(self, root, width, height, dataType='train'):
        self.width = width
        self.height = height
        self.dataType = dataType
        
        self.version = "v2.0"
        self.root = root
        
        panoptic_json_path = self.root + f"{'training'}/{self.version}/panoptic/panoptic_2020.json"
        with open(panoptic_json_path) as panoptic_file:
            panoptic = json.load(panoptic_file)
            
        self.annotations = []
        for annotation in panoptic["annotations"]:
            self.annotations.append(annotation)
            
        if self.dataType == 'train':
            self.annotations = self.annotations[:int(len(self.annotations) * 0.9)]
        elif self.dataType == 'eval':
            self.annotations = self.annotations[int(len(self.annotations) * 0.9):]
            
        print(f"{self.dataType} Number of annotations: {len(self.annotations)}")
        
        self.transform = Compose([
            ToTensor(),
            Resize((self.width, self.height), antialias=None),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):  
        return 64
        return len(self.annotations)#  // 16 * 16
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        file_name = annotation["image_id"]
        image_path = self.root + f"{'training'}/images/{file_name + '.jpg'}"
        image = Image.open(image_path)
        image = self.transform(image)
        
        segments_info = annotation["segments_info"]
        
        panoptic_path = self.root + f"{'training'}/{self.version}/panoptic/{file_name + '.png'}"
        panoptic_image = Image.open(panoptic_path)
        
        if len(segments_info) > 63:
            seg = torch.zeros(len(segments_info), self.width, self.height, dtype=torch.float32)
        else:
            seg = torch.zeros(64, self.width, self.height, dtype=torch.float32)
                    
        transform = ToTensor()
        panoptic_image = transform(panoptic_image) * 255.0
        panoptic_image = panoptic_image.to(torch.int16)
        panoptic_image = panoptic_image[0,:,:] + (2**8)*panoptic_image[1,:,:] + (2**16)*panoptic_image[2,:,:]
        panoptic_image = resize(panoptic_image.unsqueeze(0), size=(self.height, self.width), interpolation=Image.NEAREST).squeeze(0)

        for i, seg_info in enumerate(segments_info):
            seg[i] = panoptic_image == seg_info["id"]
        
        seg = seg[seg.sum(dim=(1, 2)).argsort(descending=True)]
        seg = seg[:64]
        
        image, seg = self.random_effect(image, seg)
        
        return image.cpu(), seg.cpu()

    def random_effect(self, image, instance_image):
        scale = random.uniform(1.0, 2.0)
        width = int(self.width * scale)
        height = int(self.height * scale)

        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(width, height), mode='nearest').squeeze(0)
        instance_image = torch.nn.functional.interpolate(instance_image.unsqueeze(0), size=(width, height), mode='nearest').squeeze(0)
        x = random.randint(0, width - self.width)
        y = random.randint(0, height - self.height)
        image = image[:, x:x+self.width, y:y+self.height]
        instance_image = instance_image[:, x:x+self.width, y:y+self.height]
        
        if random.randint(0, 1) == 0:
            return [image, instance_image]
        image = torch.flip(image, [2])
        instance_image = torch.flip(instance_image, [2])
        
        return image, instance_image
    
    def save_all(self):
        for i in range(len(self)):
            image, seg = self[i]
            # image랑 seg 합쳐서 torch save
            torch.save((image, seg), f"/mnt/my_passport/MapillaryPanopticSet_256/{i}.pt")

# root = '/home/wooyung/Develop/RadarDetection/CocoSeg/Mapillary/'
# train_dataset = MapillaryData(root, 256, 256)
# train_dataset.save_all()
    
    
class PreProcessedMapillaryDataset(Dataset):
    def __init__(self, root, type='train'):
        self.root = root
        # self.devices = [torch.device('cuda:0'), torch.device('cuda:1')]
        
        self.type = type
        self.data = []
        for f in os.listdir(self.root):
            if f.endswith('.pt'):
                self.data.append(f)
                
        self.data.sort()
                
        if self.type == 'train':
            self.data = self.data[:int(len(self.data) * 0.9)]
        elif self.type == 'eval':
            self.data = self.data[int(len(self.data) * 0.9):]
            
        print(f"{type} Number of images: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, seg = torch.load(os.path.join(self.root, self.data[idx]))
        if self.type == 'train':
            image, seg = self.random_effect(image, seg)
        return image, seg

    def random_effect(self, image, instance_image):
        origin_width = image.shape[1]
        origin_height = image.shape[2]
        
        scale = random.uniform(1.0, 1.2)
        width = int(origin_width * scale)
        height = int(origin_height * scale)

        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(width, height), mode='nearest').squeeze(0)
        instance_image = torch.nn.functional.interpolate(instance_image.unsqueeze(0), size=(width, height), mode='nearest').squeeze(0)
        x = random.randint(0, width - origin_width)
        y = random.randint(0, height - origin_height)
        image = image[:, x:x+origin_width, y:y+origin_height]
        instance_image = instance_image[:, x:x+origin_width, y:y+origin_height]
        
        if random.randint(0, 1) == 0:
            return [image, instance_image]
        image = torch.flip(image, [2])
        instance_image = torch.flip(instance_image, [2])
        
        return image, instance_image