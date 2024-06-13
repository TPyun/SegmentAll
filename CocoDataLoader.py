from hmac import new
import os
from turtle import back
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
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from sympy import E


class NewSegDataset(torch.utils.data.Dataset):
    def __init__(self, path, width, height, mode='train', random=False, num_masks=64, num_classes=4):
        self.width = width
        self.height = height

        self.mode = mode
        self.random = random
        self.data_list = []
        
        self.num_classes = num_classes
        self.num_masks = num_masks

        for folder in os.listdir(path):
            if not os.path.isdir(os.path.join(path, folder)):
                continue
            
            # 각각의 case 열기
            for sub_folder in os.listdir(os.path.join(path, folder)):
                if 'task' not in sub_folder:
                    continue
                task_json_path = osp.join(path, folder, sub_folder, 'task.json')
                with open(task_json_path, 'r') as task_json_file:
                    task_json_data = json.load(task_json_file)
                if task_json_data['name'] != 'FrontCam': # and task_json_data['name'] != 'FisheyeCam':
                    continue
                
                ann_json_name = osp.join(path, folder, sub_folder, 'annotations.json')
                for file in os.listdir(os.path.join(path, folder, sub_folder)):
                    
                    if 'data' in file:
                        # 여기서 이미지 목록이랑 사이즈 가져옴
                        mani_json_name = 'manifest.jsonl'  
                        
                        if not os.path.exists(osp.join(path, folder, sub_folder, file, mani_json_name)):
                            continue
                        
                        image_list = []
                        with open(osp.join(path, folder, sub_folder, file, mani_json_name), 'r') as mani_json_file:
                            for line in mani_json_file:
                                mani_json_line = json.loads(line)
                                try:
                                    # print(osp.join(path, folder, sub_folder, file, mani_json_line['name'] + ".jpg"))
                                    image_list.append(osp.join(path, folder, sub_folder, file, mani_json_line['name'] + ".jpg"))
                                except Exception as e:
                                    continue
                                
                        anno_dict = {ann_json_name: image_list}
                        self.data_list.append(anno_dict)

        num_cases = len(self.data_list)
        
        # if mode == 'train':
        #     self.data_list = self.data_list[:int(num_cases * 0.9)]
        # elif mode == 'test':
        #     self.data_list = self.data_list[int(num_cases * 0.1):]
        # else:
        #     raise ValueError("Invalid mode")
        
        self.num_images = 0
        for case in self.data_list:
            for key in case.keys():
                self.num_images += len(case[key])
        print(f"Number of cases: {num_cases}, Number of images: {self.num_images}")
        
        
    def __len__(self):
        return self.num_images

    def get_label_id(self, label):
        if label == 'Wall':
            return 1
        elif label == 'Pedestrian':
            return 2
        elif label == 'Vehicle':
            return 3
        else:
            print(f"Invalid label: {label}")
            
    def __getitem__(self, index):
        case_idx = 0
        image_idx = 0
        for case in self.data_list:
            for key in case.keys():
                if image_idx + len(case[key]) > index:
                    break
                image_idx += len(case[key])
                case_idx += 1
            if image_idx + len(case[key]) > index:
                break
        image_idx = index - image_idx
        
        ann_json_path = key
        image_path = case[key][image_idx]
        
        image = Image.open(image_path)
        original_image_size = image.size
        image = image.resize((self.width, self.height))
        image = transforms.ToTensor()(image)
        
        with open(ann_json_path, 'r') as ann_json_file:
            ann_json_data = json.load(ann_json_file)
            
        instance_seg = torch.zeros((self.num_masks, self.width, self.height), dtype=torch.float32)
        label_list = torch.zeros((self.num_masks, self.num_classes), dtype=torch.float32)
        background = torch.ones((self.width, self.height), dtype=torch.float32)
        
        for main_data in ann_json_data:
            mask_iter = 1
            for sub_data in main_data['shapes']:
                if sub_data['frame'] != image_idx:
                    continue
                
                # mask의 클래스
                label = sub_data['label']
                class_softmax = torch.zeros(self.num_classes)
                class_softmax[self.get_label_id(label)] = 1
                label_list[mask_iter] = class_softmax
                
                # 포인트 xy변환하고 세그 이미지에 넣기
                points = sub_data['points']
                processed_point = []
                for x, y in zip(points[::2], points[1::2]):
                    processed_point.append([x, y])
                    
                processed_point = np.array(processed_point, np.int32)
                
                # 해당 라벨에 대한 채널만을 위한 마스크 생성
                mask = np.zeros((original_image_size[1], original_image_size[0]), np.uint8)
                cv2.fillPoly(mask, [processed_point], (1))
                mask = cv2.resize(mask, (self.width, self.height))
                
                instance_seg[mask_iter] = torch.where(torch.tensor(mask, dtype=torch.float32) == 1, 1, instance_seg[mask_iter])
                mask_iter += 1
                
                background[mask == 1] = 0
        
        instance_seg[0] = background
        class_softmax = torch.zeros(self.num_classes)
        class_softmax[0] = 1
        label_list[0] = class_softmax
        
        if self.random == True:
            image, instance_seg = self.random_effect(image, instance_seg)
            
        for i in range(self.num_masks):
            if instance_seg[i].sum() == 0:
                label_list[i] = torch.zeros(self.num_classes)
                
        return image, instance_seg, label_list           

    def random_effect(self, image, instance_image):
        # 색상 조정
        color_jitter = transforms.ColorJitter(brightness=(0.1, 2.0),contrast=(0.1, 2.0),saturation=(0.1, 2.0))
        image = color_jitter(image)
        
        # 랜덤으로 좌우 반전
        random_direction = random.randint(0, 1)
        if random_direction == 1:
            image = torch.flip(image, [2])
            instance_image = torch.flip(instance_image, [2])
        
        # 왜곡효과
        startpoints = [[0, 0], [255, 0], [255, 255], [0, 255]]
        gap_x = self.width
        gap_y = self.height
        endpoints = [[-random.randint(0, gap_x), -random.randint(0, gap_y)], \
            [random.randint(255, 255+gap_x), -random.randint(0, gap_y)], \
                [random.randint(255, 255+gap_x), random.randint(255, 255+gap_y)], \
                    [-random.randint(0, gap_x), random.randint(255, 255+gap_y)]]
        
        image = TF.perspective(image, startpoints, endpoints)
        instance_image = TF.perspective(instance_image, startpoints, endpoints)
        
        # # 스케일 조정
        # scale = random.uniform(1.0, 2.0)
        # width = int(self.width * scale)
        # height = int(self.width * scale)
        # image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
        # instance_image = torch.nn.functional.interpolate(instance_image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
        # x = random.randint(0, width - self.width)
        # y = random.randint(0, height - self.height)
        # image = image[:, x:x+self.width, y:y+self.height]
        # instance_image = instance_image[:, x:x+self.width, y:y+self.height]
        
        return image, instance_image


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
    def __init__(self, root, dataType, width, height, num_masks=64, num_classes=4):
        self.root = root
        self.dataType = dataType
        self.image_folder = '{}/{}'.format(self.root, dataType)
        self.annotation_file = '{}/panoptic_annotations/panoptic_{}.json'.format(self.root, dataType)
        self.panoptic_image_folder = '{}/panoptic_annotations/panoptic_{}/'.format(self.root, dataType)
        self.image_list = os.listdir(self.image_folder)
        self.image_list.sort()
        
        self.width = width
        self.height = height
        
        self.num_masks = num_masks
        self.num_classes = num_classes
        
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
        
        instance_seg = torch.zeros((self.num_masks, self.width, self.height), dtype=torch.float32)
        label_list = torch.zeros((self.num_masks, self.num_classes), dtype=torch.float32)

        # 고유 ID를 사용하여 객체별 세그멘테이션 정보 추출
        for i, segment_info in enumerate(image_info['segments_info']):
            if i > 63:
                break
            
            segment_id = segment_info['id']
            category_id = segment_info['category_id']
            
            class_softmax = torch.zeros(self.num_classes)
            class_softmax[category_id] = 1
            label_list[i] = class_softmax
                
            segment_mask = panoptic_id == segment_id
            segment_mask = segment_mask.astype(np.float32)
            segment_mask = cv2.resize(segment_mask, (self.width, self.height))
            mask = torch.tensor(segment_mask, dtype=torch.float32)
            instance_seg[i] = torch.where(mask == 1, 1, instance_seg[i])

        # if 'train' in self.dataType:
        #     image, instance_seg = self.random_effect(image, instance_seg)
            
        for i in range(64):
            if instance_seg[i].sum() == 0:
                label_list[i] = torch.zeros(self.num_classes)
    
        return image, instance_seg, label_list
    
    def __len__(self):
        return 64
        # if 'train' in self.dataType:
        #     return 16000
        # elif 'val' in self.dataType:
        #     return 1024
        return self.length // 4 * 4

    
    def random_effect(self, image, instance_image):
        color_jitter = transforms.ColorJitter(brightness=(0.5, 1.5),contrast=(0.5, 1.5),saturation=(0.5, 1.5))
        image = color_jitter(image)
        
        scale = random.uniform(1.0, 1.2)
        width = int(self.width * scale)
        height = int(self.width * scale)

        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
        instance_image = torch.nn.functional.interpolate(instance_image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
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
        return 32
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        file_name = annotation["image_id"]
        image_path = self.root + f"{'training'}/images/{file_name + '.jpg'}"
        image = Image.open(image_path)
        image = self.transform(image)
        
        segments_info = annotation["segments_info"]
        
        panoptic_path = self.root + f"{'training'}/{self.version}/panoptic/{file_name + '.png'}"
        panoptic_image = Image.open(panoptic_path)
        
        # if len(segments_info) > 63:
        #     seg = torch.zeros(len(segments_info), self.width, self.height, dtype=torch.float32)
        # else:
        #     seg = torch.zeros(64, self.width, self.height, dtype=torch.float32)
            
        seg = torch.zeros(1, self.width, self.height, dtype=torch.float32)
        
        transform = ToTensor()
        panoptic_image = transform(panoptic_image) * 255.0
        panoptic_image = panoptic_image.to(torch.int16)
        panoptic_image = panoptic_image[0,:,:] + (2**8)*panoptic_image[1,:,:] + (2**16)*panoptic_image[2,:,:]
        panoptic_image = resize(panoptic_image.unsqueeze(0), size=(self.height, self.width), interpolation=Image.NEAREST).squeeze(0)

        for i, seg_info in enumerate(segments_info):
            # seg[i] = panoptic_image == seg_info["id"]
            
            mask = panoptic_image == seg_info["id"]
            edges = cv2.Canny(mask.numpy().astype(np.uint8), 0, 1)
            edges = torch.tensor(edges, dtype=torch.float32)
            edges = edges.unsqueeze(0)
            edges = torch.where(edges > 0.5, 1, 0)
            seg = torch.where(edges == 1, 1, seg)
            
        # plt.clf()
        # plt.figure(figsize=(12, 12))
        # plt.imshow(seg.permute(1, 2, 0).numpy())
        # plt.savefig('/home/wooyung/Develop/RadarDetection/seg.png', bbox_inches='tight')
        # plt.close()
            
        
        # seg = seg[seg.sum(dim=(1, 2)).argsort(descending=True)]
        # seg = seg[:64]
        
        # output_mask = torch.zeros(self.width, self.height, dtype=torch.float32)
        # for i, s in enumerate(seg):
        #     output_mask += (s * (i + 1)  * 10)
        # output_mask = (output_mask - output_mask.min()) / (output_mask.max() - output_mask.min())
            
        # output_mask = output_mask.unsqueeze(0)
        
        # image, seg = self.random_effect(image, seg)
        
        return image, seg

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
    def __init__(self, root, width, height, type='train'):
        self.root = root
        self.width = width
        self.height = height
        self.resize = Resize((self.width, self.height), interpolation=Image.NEAREST)
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
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
        
        image = self.resize(image)
        seg = self.resize(seg)
        
        if self.type == 'train':
            image, seg = self.random_effect(image, seg)
            
        image_mean_brightness = image.mean()
        image = image - image_mean_brightness + 0.5
        image = self.normalize(image)
        
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
    
    

class BorderPanopticCocoDataset(Dataset):
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
        
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
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

        line_image = torch.zeros((1, self.width, self.height), dtype=torch.float32)
        
        if len(image_info['segments_info']) > 63:
            instance_seg = torch.zeros((len(image_info['segments_info']), self.width, self.height), dtype=torch.float32)
        else:
            instance_seg = torch.zeros((64, self.width, self.height), dtype=torch.float32)

        # 고유 ID를 사용하여 객체별 세그멘테이션 정보 추출
        for i, segment_info in enumerate(image_info['segments_info']):
            segment_id = segment_info['id']
            category_id = segment_info['category_id']
            
            # line seg 만들기
            segment_mask = torch.tensor(panoptic_id == segment_id, dtype=torch.float32)
            segment_mask = F.interpolate(segment_mask.unsqueeze(0).unsqueeze(0), size=(self.width, self.height), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
            segment_mask = torch.where(segment_mask > 0.5, 1, 0)
            segment_mask = segment_mask.numpy() 
            
            edges = cv2.Canny(segment_mask.astype(np.uint8), 0, 1)
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            edges = torch.tensor(edges, dtype=torch.float32)
            edges = torch.where(edges > 0.5, 1, 0)
            line_image = torch.where(edges == 1, 1, line_image)
            
            # 객체별 seg만들기
            segment_mask = panoptic_id == segment_id
            segment_mask = segment_mask.astype(np.float32)
            mask = torch.tensor(segment_mask, dtype=torch.float32)
            mask = mask.unsqueeze(0)
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(self.width, self.height), mode='bilinear', align_corners=True).squeeze(0)
            instance_seg[i] = torch.where(mask == 1, 1, instance_seg[i])
            
        line_image = (line_image == 0).float()
        
        
        instance_seg = instance_seg[instance_seg.sum(dim=(1, 2)).argsort(descending=True)]
        instance_seg = instance_seg[:64]
        
        
        # if 'train' in self.dataType:
        #     image, line_image = self.random_effect(image, line_image)
        
        image_mean_brightness = image.mean()
        image = image - image_mean_brightness + 0.5
        image = self.normalize(image)
    
        return image, line_image, instance_seg
    
    def __len__(self):
        # if 'train' in self.dataType:
        #     return 8192
        # elif 'val' in self.dataType:
        #     return 1024
        return self.length // 4 * 4

    
    def random_effect(self, image, line_image, seg_image):
        color_jitter = transforms.ColorJitter(brightness=(0.5, 1.5),contrast=(0.5, 1.5),saturation=(0.5, 1.5))
        image = color_jitter(image)
        
        scale = random.uniform(1.0, 1.5)
        width = int(self.width * scale)
        height = int(self.width * scale)

        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
        line_image = torch.nn.functional.interpolate(line_image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
        seg_image = torch.nn.functional.interpolate(seg_image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
        x = random.randint(0, width - self.width)
        y = random.randint(0, height - self.height)
        image = image[:, x:x+self.width, y:y+self.height]
        line_image = line_image[:, x:x+self.width, y:y+self.height]
        seg_image = seg_image[:, x:x+self.width, y:y+self.height]
        
        flip_seed = random.randint(0, 3)
        if flip_seed == 0:
            return [image, line_image, seg_image]
        elif flip_seed == 1:
            image = torch.flip(image, [2])
            line_image = torch.flip(line_image, [2])
            seg_image = torch.flip(seg_image, [2])
        elif flip_seed == 2:
            image = torch.flip(image, [1])
            line_image = torch.flip(line_image, [1])
            seg_image = torch.flip(seg_image, [1])
        else:
            image = torch.flip(image, [1, 2])
            line_image = torch.flip(line_image, [1, 2])
            seg_image = torch.flip(seg_image, [1, 2])
        
        return image, line_image, seg_image


class BorderPreProcessedMapillaryDataset(Dataset):
    def __init__(self, root, width, height, type='train'):
        self.root = root
        self.width = width
        self.height = height
        self.resize = Resize((self.width, self.height), interpolation=Image.NEAREST)
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
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
        return 32
        return len(self.data)
    
    def __getitem__(self, idx):
        image, seg = torch.load(os.path.join(self.root, self.data[idx]))
        
        image = self.resize(image)
        seg = self.resize(seg)
        
        border_image = torch.zeros((1, self.width, self.height), dtype=torch.float32)
        
        for i in range(1, 64):
            edges = cv2.Canny(seg[i].numpy().astype(np.uint8), 0, 1)
            kernel = np.ones((3, 3), np.uint8)  # 3x3 크기의 사각형 커널
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            edges = torch.tensor(edges, dtype=torch.float32)
            edges = edges.unsqueeze(0)
            edges = F.interpolate(edges.unsqueeze(0), size=(self.width, self.height), mode='bilinear', align_corners=True).squeeze(0)
            edges = torch.where(edges > 0.5, 1, 0)
            border_image = torch.where(edges == 1, 1, border_image)
        
        # if self.type == 'train':
        #     image, seg = self.random_effect(image, seg)
            
        image_mean_brightness = image.mean()
        image = image - image_mean_brightness + 0.5
        image = self.normalize(image)
        
        return image, border_image

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
    
    
    
    
class OneMaskPanopticCocoDataset(Dataset):
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
        
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        
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

        border_image = torch.zeros((1, self.width, self.height), dtype=torch.float32)
        instance_seg = torch.zeros((1, self.width, self.height), dtype=torch.float32)

        # 고유 ID를 사용하여 객체별 세그멘테이션 정보 추출
        for i, segment_info in enumerate(image_info['segments_info']):
            segment_id = segment_info['id']
            category_id = segment_info['category_id']
            
            segment_mask = torch.tensor(panoptic_id == segment_id, dtype=torch.float32)
            segment_mask = F.interpolate(segment_mask.unsqueeze(0).unsqueeze(0), size=(self.width, self.height), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
            segment_mask = torch.where(segment_mask > 0.5, 1, 0)
            
            numpy_segment_mask = segment_mask.numpy() 
            edges = cv2.Canny(numpy_segment_mask.astype(np.uint8), 0, 1)
            kernel = np.ones((1, 1), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = torch.tensor(edges, dtype=torch.float32)
            edges = torch.where(edges > 0.5, 1, 0)
            border_image = torch.where(edges == 1, 1, border_image)
            
            instance_seg = torch.where(segment_mask == 1, category_id / 255, instance_seg)
        
        instance_seg = torch.where(border_image == 1, 0, instance_seg)
        plt.clf()
        plt.figure(figsize=(12, 12))
        plt.imshow(instance_seg.squeeze(0).numpy())
        plt.savefig('/home/wooyung/Develop/RadarDetection/seg.png', bbox_inches='tight')
        plt.close()
            
        # image, border_image = self.random_effect(image, border_image)

        image_mean_brightness = image.mean()
        image = image - image_mean_brightness + 0.5
        image = self.normalize(image)
    
        return image, instance_seg
    
    def __len__(self):
        return 16
        if 'train' in self.dataType:
            return 8192
        elif 'val' in self.dataType:
            return 1024
        return self.length // 4 * 4

    
    def random_effect(self, image, instance_image):
        color_jitter = transforms.ColorJitter(brightness=(0.5, 1.5),contrast=(0.5, 1.5),saturation=(0.5, 1.5))
        image = color_jitter(image)
        
        scale = random.uniform(1.0, 1.5)
        width = int(self.width * scale)
        height = int(self.width * scale)

        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
        instance_image = torch.nn.functional.interpolate(instance_image.unsqueeze(0), size=(width, height), mode='bilinear', align_corners=True).squeeze(0)
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