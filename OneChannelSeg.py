from turtle import color
from cv2 import mean
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time
import torch.nn.functional as F
from pycocotools import cocoeval
import numpy as np
import sys
import cv2
import random

import ImageProcess as ip
import Model as md
import CocoDataLoader as ccdl
import Tele

def get_matching_channel_list(pred, target, loss_func='smooth'):
    # 3차원 이미지용
    num_pred = pred.size(0)
    num_target = target.size(0)
    target = (target > 0.5).float()
    target_sums = target.sum(dim=(1, 2))

    # 손실 함수 선택
    if loss_func == 'smooth':
        loss_fn = F.smooth_l1_loss
    elif loss_func == 'mse':
        loss_fn = F.mse_loss
    elif loss_func == 'bce':
        loss_fn = F.binary_cross_entropy

    loss_list = []
    # 손실 계산 최적화
    for p in range(num_pred):
        for t in range(num_target):
            if target_sums[t].item() == 0:
                continue
            loss = loss_fn(pred[p], target[t], reduction='mean')
            loss_list.append([p, t, loss.item()])
            
    loss_list.sort(key=lambda x: x[2])
        
    pred_matched, target_matched = set(), set()
    match_list = []
    for p, t, loss in loss_list:
        if p not in pred_matched and t not in target_matched:
            match_list.append([p, t, loss])
            pred_matched.add(p)
            target_matched.add(t)
    return match_list
    
def rearrange_target_image(pred, target):
    new_target = torch.zeros_like(pred)
    for b in range(pred.shape[0]):
        match_list = get_matching_channel_list(pred[b], target[b])
        # pred에 맞게 target의 채널을 변경
        new_target_batch = torch.zeros_like(pred[b])
        for match in match_list:
            new_target_batch[match[0]] = target[b][match[1]]
        new_target[b] = new_target_batch
    return new_target

def loss_each_channel(pred, target, loss_func):
    if loss_func == 'smooth':
        loss = F.smooth_l1_loss(pred, target, reduction='none').mean(dim=(2, 3))
    elif loss_func == 'mse':
        loss = F.mse_loss(pred, target, reduction='none').mean(dim=(2, 3))
    elif loss_func == 'bce':
        loss = F.binary_cross_entropy(pred, target, reduction='none').mean(dim=(2, 3))
    elif loss_func == 'bce_logit':
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none').mean(dim=(2, 3))
    elif loss_func == 'iou':
        smooth = 1e-6
        intersection = (pred * target).float().sum(dim=(2, 3))
        union = (pred + target).float().sum(dim=(2, 3))
        iou = intersection / (union + smooth)
        loss = 1 - iou
    elif loss_func == 'dice':
        smooth = 1e-6
        inverse_pred = 1 - pred
        inverse_target = 1 - target

        intersection = (inverse_pred * inverse_target).float().sum(dim=(2, 3))
        union = (inverse_pred + inverse_target).float().sum(dim=(2, 3))

        dice = (2 * intersection + smooth) / (union + smooth)
        loss = 1 - dice
            
    mean_loss = loss.mean()
    return mean_loss

def get_iou(pred, target):
    pred = pred > 0.5
    target = target > 0.5
    
    total_iou = []
    total_except_error_iou = []
    for b in range(pred.shape[0]):
        batch_iou = []
        batch_except_error_iou = []
        
        for c in range(pred.shape[1]):
            intersection = (pred[b, c] & target[b, c]).float().sum()
            union = (pred[b, c] | target[b, c]).float().sum()
            # 둘다 0인 경우는 제외
            if union == 0:
                continue
            iou = intersection / union
            batch_iou.append(iou)
            
            if target[b, c].sum() == 0:
                continue
            batch_except_error_iou.append(iou)
            
        if len(batch_iou) == 0:
            continue
        total_iou.append(sum(batch_iou) / len(batch_iou))
        if len(batch_except_error_iou) == 0:
            continue
        total_except_error_iou.append(sum(batch_except_error_iou) / len(batch_except_error_iou))
        
    if len(total_iou) == 0:
        a = torch.tensor(0.0)
    else:
        a = sum(total_iou) / len(total_iou)
    if len(total_except_error_iou) == 0:
        b = torch.tensor(0.0)
    else:
        b = sum(total_except_error_iou) / len(total_except_error_iou)
        
    return a, b

def get_loss(pred, target, loss_func='smooth'):
    mask_loss = loss_each_channel(pred, target, loss_func)
    return mask_loss

def eval(model, eval_dataloader, device):
    global eval_plot_points, eval_iter
    
    model.eval()
    
    total_loss = 0
    total_iou = 0
    total_except_error_iou = 0
    epoch_start_time = time.time()
    for batch_i, batch in enumerate(eval_dataloader):
        image = batch[0].to(device)
        line_image = batch[1].to(device)
        seg_image = batch[2].to(device)

        with torch.no_grad():
            result = model(image)
        result = result.detach()
        multi_channel_result = split_channels(result[0][0]).unsqueeze(0).to(device)
        
        # plt.clf()
        # plt.figure(figsize=(12, 12))
        # for i in range(multi_channel_result.shape[1]):
        #     plt.subplot(8, 8, i + 1)
        #     plt.subplots_adjust(wspace=0, hspace=0)
        #     plt.axis('off')
        #     plt.imshow(multi_channel_result[0][i].cpu().numpy())
        # plt.savefig(f'{directory}/Eval/instance_mask.png', bbox_inches='tight', pad_inches=0)
        # plt.close()
            
            
        # infer_end = time.time()
        # print(f"Inference Time: {infer_end - infer_start:.8f}")
        
        loss = get_loss(result, line_image)
        re_seg_image = rearrange_target_image(multi_channel_result, seg_image)
        iou, except_error_iou = get_iou(multi_channel_result, re_seg_image)
        # iou, except_error_iou = torch.tensor(0.0), torch.tensor(0.0)
        # print(iou, except_error_iou)
        
        # plt.clf()
        # plt.figure(figsize=(12, 12))
        # for i in range(20):
        #     try:
        #         plt.subplot(4, 5, i + 1)
        #         plt.subplots_adjust(wspace=0, hspace=0)
        #         plt.axis('off')
        #         plt.imshow(multi_channel_result[0][i].cpu().numpy())
        #     except Exception as e:
        #         break
        # plt.savefig(f'{directory}/Train/instance.png', bbox_inches='tight', pad_inches=0)
        # plt.close()
        # for i in range(20):
        #     try:
        #         plt.subplot(4, 5, i + 11)
        #         plt.subplots_adjust(wspace=0, hspace=0)
        #         plt.axis('off')
        #         plt.imshow(re_seg_image[0][i].cpu().numpy())
        #     except Exception as e:
        #         break
        # plt.savefig(f'{directory}/Eval/instance.png', bbox_inches='tight', pad_inches=0)
        # plt.close()

        
        plt.clf()
        plt.imshow(ip.classes_to_rand_rgb(multi_channel_result[0]).detach().permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.savefig(f'{directory}/Eval/instance_mask.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        plt.clf()
        plt.imshow(ip.classes_to_rand_rgb(seg_image[0]).detach().permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.savefig(f'{directory}/Eval/seg_image.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # print(f"IoU: {iou:.8f} Except Error IoU: {except_error_iou:.8f}")

        total_loss += loss.item()
        total_iou += iou.item()
        total_except_error_iou += except_error_iou.item()
        
        progress = (batch_i + 1) / len(eval_dataloader)
        bar_length = 50
        bar = "#" * int(bar_length * progress) + "-" * (bar_length - int(bar_length * progress))

        elapsed_time = time.time() - epoch_start_time
        epoch_time_estimate = elapsed_time / (batch_i + 1) * len(eval_dataloader)
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(epoch_time_estimate))
        
        sys.stdout.write(f'\r[{bar}] {progress * 100:.2f}% Loss:{loss.item():.4f} ET:{formatted_time}')
        sys.stdout.flush()
        
    total_loss /= len(eval_dataloader)
    total_iou /= len(eval_dataloader)
    total_except_error_iou /= len(eval_dataloader)
    
    print()
    print(f"Eval Loss: {total_loss:.8f}", end='| ')
    print(f"Eval IoU: {total_iou:.8f}", end='| ')
    print(f"Eval Except Error IoU: {total_except_error_iou:.8f}", end='| ')
    print()
    eval_plot_points.append([eval_iter, total_loss, total_iou, total_except_error_iou])
    eval_iter += 1     
    
    # plt.clf()
    # plt.figure(figsize=(12, 12))
    # for i in range(3):
    #     try:
    #         plt.subplot(4, 3, 3 * i + 1)
    #         plt.subplots_adjust(wspace=0, hspace=0)
    #         plt.axis('off')
    #         plt.imshow(np.clip(image[i].permute(1, 2, 0).cpu().numpy(), 0, 1))
    #         plt.subplot(4, 3, 3 * i + 2)
    #         plt.subplots_adjust(wspace=0, hspace=0)
    #         plt.axis('off')
    #         plt.imshow(result[i].detach().permute(1, 2, 0).cpu().numpy()) 
    #         plt.subplot(4, 3, 3 * i + 3)
    #         plt.subplots_adjust(wspace=0, hspace=0)
    #         plt.axis('off')
    #         plt.imshow(line_image[i].detach().permute(1, 2, 0).cpu().numpy())
    #     except Exception as e:
    #         break
    # plt.savefig(f'{directory}/Eval/instance.png', bbox_inches='tight', pad_inches=0)
    # plt.close()
    
    plt.clf()
    plt.figure(figsize=(12, 12))
    for point in eval_plot_points:
        plt.plot(point[0], point[1], 'ro')
    plt.savefig(f'{directory}/Eval/loss.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    plt.clf()
    plt.figure(figsize=(12, 12))
    for point in eval_plot_points:
        plt.plot(point[0], point[2], 'ro')
    plt.savefig(f'{directory}/Eval/iou.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    plt.clf()
    plt.figure(figsize=(12, 12))
    for point in eval_plot_points:
        plt.plot(point[0], point[3], 'ro')
    plt.savefig(f'{directory}/Eval/except_error_iou.png', bbox_inches='tight', pad_inches=0)
    plt.close()
        
    model.train()
    
def split_channels(line_image, threshold=0.92):
    line_image = line_image > threshold
    line_image = line_image.cpu().numpy()
    line_image = line_image.astype(np.uint8)
    line_image = line_image * 255
    
    num_labels, labels_im = cv2.connectedComponents(line_image)
    # 각 레이블에 대해 이미지 생성
    channels = torch.zeros((num_labels - 1, line_image.shape[0], line_image.shape[1]), dtype=torch.float32)
    for label in range(1, num_labels):  # 배경(레이블 0)은 무시
        mask = labels_im == label
        channels[label - 1] = torch.tensor(mask, dtype=torch.float32)
        
    return channels
        
script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)
os.chdir(directory)

width = 512
height = 512
actual_batch_size = 4
train_plot_points = []
eval_plot_points = []
eval_iter = 0

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank} initialized.")
    
def cleanup():
    dist.destroy_process_group()
    
def main(rank, world_size):
    print(f"Running DDP example on rank {rank}.")
    setup(rank, world_size)
    device = torch.device("cuda", rank)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    
    model = md.CustomDeepLabV3().to(device)
    model_name = model.__class__.__name__
    model_path = f'{directory}/Model/translator_{model_name}.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=map_location))
        print(f"Loaded {model_path}!")
    else:   
        print(f"Model {model_path} not exists.")
    ddp_model = DDP(model, device_ids=[rank])
    
    optimizer = optim.NAdam(ddp_model.parameters(), lr=0.00005)

    root = f'{directory}/COCO'
    train_dataset = ccdl.BorderPanopticCocoDataset(root, 'train2017', width, height)
    
    # root = f'{directory}/SAMDataset'
    # train_dataset = ccdl.SAMDataset(root, 'train', width, height)
    
    # root = '/home/wooyung/Develop/RadarDetection/20240404/'
    # train_dataset = ccdl.SegDataset(root, width, height)
    
    # root = '/home/wooyung/Develop/RadarDetection/SegmentAll/Mapillary/'
    # train_dataset = ccdl.MapillaryData(root, width, height)
    
    # root = '/home/wooyung/Develop/RadarDetection/SegmentAll/MapillaryPanopticSet_256/'
    # train_dataset = ccdl.BorderPreProcessedMapillaryDataset(root, width, height, type='train')

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=actual_batch_size, sampler=train_sampler, pin_memory=True, num_workers=3)
    
    if rank == 0:
        eval_dataset = ccdl.BorderPanopticCocoDataset(root, 'val2017', width, height)
        # eval_dataset = ccdl.BorderPreProcessedMapillaryDataset(root, width, height, type='eval')
        # eval_dataset = ccdl.MapillaryData(root, width, height, 'eval')
        # eval_dataset = ccdl.SegDataset(root, width, height, mode='eval')
        # eval_dataset = ccdl.SAMDataset(root, 'val', width, height)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=True)
        
    
    num_epochs = 100000
    print_every = 1
    eval_every = 4
    train = True
    
    t_loss_epoch = 0
    t_iou_epoch = 0
    t_except_error_iou_epoch = 0
    
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    
    for epoch in range(num_epochs):
        if train:
            train_sampler.set_epoch(epoch)
            
            loss_epoch = 0
            iou_epoch = 0
            except_error_iou_epoch = 0
            
            epoch_start_time = time.time()
            for batch_i, batch in enumerate(train_dataloader):
                image = batch[0].to(device)
                line_image = batch[1].to(device)

                # 객체 탐지 모델 학습
                result = ddp_model(image)
                
                loss = get_loss(result, line_image)
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                if rank == 0:
                    result = result.detach()
                    # result = result > 0.5
                    
                    loss_epoch += loss.item()
                    
                    # seg_image = batch[2]
                    # multi_channel_result = split_channels(result[0][0]).unsqueeze(0).to(device)
                    # re_seg_image = rearrange_target_image(multi_channel_result, seg_image)
                    # iou, except_error_iou = get_iou(multi_channel_result, re_seg_image)
                    iou, except_error_iou = torch.tensor(0.0), torch.tensor(0.0)
                    
                    
                    
                    # plt.clf()
                    # plt.imshow(ip.classes_to_rand_rgb(multi_channel_result[0]).detach().permute(1, 2, 0).cpu().numpy())
                    # plt.axis('off')
                    # plt.savefig(f'{directory}/Train/instance_mask.png', bbox_inches='tight', pad_inches=0)
                    # plt.close()
                    
                    # plt.clf()
                    # plt.imshow(ip.classes_to_rand_rgb(seg_image[0]).detach().permute(1, 2, 0).cpu().numpy())
                    # plt.axis('off')
                    # plt.savefig(f'{directory}/Train/seg_image.png', bbox_inches='tight', pad_inches=0)
                    # plt.close()
                    
                    iou_epoch += iou.item()
                    except_error_iou_epoch += except_error_iou.item()
                    
                
                    progress = (batch_i + 1) / len(train_dataloader)
                    bar_length = 50
                    bar = "#" * int(bar_length * progress) + "-" * (bar_length - int(bar_length * progress))
                    
                    elapsed_time = time.time() - epoch_start_time
                    epoch_time_estimate = elapsed_time / (batch_i + 1) * len(train_dataloader)
                    formatted_time = time.strftime("%H:%M:%S", time.gmtime(epoch_time_estimate))
                    
                    sys.stdout.write(f'\r[{bar}] {progress * 100:.2f}% Loss:{loss.item():.4f} ET:{formatted_time}')
                    sys.stdout.flush()
                    
            loss_epoch /= len(train_dataloader)
            t_loss_epoch += loss_epoch
            
            iou_epoch /= len(train_dataloader)
            t_iou_epoch += iou_epoch
            
            except_error_iou_epoch /= len(train_dataloader)
            t_except_error_iou_epoch += except_error_iou_epoch
            
            if (epoch + 1) % print_every == 0:
                if rank == 0:
                    print()
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", end='| ')
                    print(f"epoch: {epoch}", end='| ')
                    print(f"loss: {t_loss_epoch / print_every:.8f}", end='| ')
                    print(f"IoU: {t_iou_epoch / print_every:.8f}", end='| ')
                    print(f"EEI: {t_except_error_iou_epoch / print_every:.8f}", end='| ')
                    print()
                    train_plot_points.append([epoch, t_loss_epoch / print_every, t_iou_epoch / print_every, t_except_error_iou_epoch / print_every])
                t_loss_epoch = 0
                t_iou_epoch = 0
                t_except_error_iou_epoch = 0
                
                if rank == 0:
                    plt.clf()
                    plt.figure(figsize=(12, 12))
                    for i in range(3):
                        try:
                            plt.subplot(4, 3, 3 * i + 1)
                            plt.subplots_adjust(wspace=0, hspace=0)
                            plt.axis('off')
                            plt.imshow(np.clip(image[i].permute(1, 2, 0).cpu().numpy(), 0, 1))
                            plt.subplot(4, 3, 3 * i + 2)
                            plt.subplots_adjust(wspace=0, hspace=0)
                            plt.axis('off')
                            plt.imshow(result[i].permute(1, 2, 0).cpu().numpy()) 
                            plt.subplot(4, 3, 3 * i + 3)
                            plt.subplots_adjust(wspace=0, hspace=0)
                            plt.axis('off')
                            plt.imshow(line_image[i].permute(1, 2, 0).cpu().numpy())
                        except Exception as e:
                            break
                    plt.savefig(f'{directory}/Train/instance.png', bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    plt.clf()
                    plt.figure(figsize=(12, 12))
                    for point in train_plot_points:
                        plt.plot(point[0], point[1], 'ro')
                    plt.savefig(f'{directory}/Train/loss.png', bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    # plt.clf()
                    # plt.figure(figsize=(12, 12))
                    # for point in train_plot_points:
                    #     plt.plot(point[0], point[2], 'ro')
                    # plt.savefig(f'{directory}/Train/iou.png', bbox_inches='tight', pad_inches=0)
                    # plt.close()
                    
                    # plt.clf()
                    # plt.figure(figsize=(12, 12))
                    # for point in train_plot_points:
                    #     plt.plot(point[0], point[3], 'ro')
                    # plt.savefig(f'{directory}/Train/except_error_iou.png', bbox_inches='tight', pad_inches=0)
                    # plt.close()
                    
                    multi_channel_result = split_channels(result[0][0]).unsqueeze(0).to(device)
                    plt.clf()
                    plt.imshow(ip.classes_to_rand_rgb(multi_channel_result[0]).detach().permute(1, 2, 0).cpu().numpy())
                    plt.axis('off')
                    plt.savefig(f'{directory}/Train/instance_mask.png', bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    seg_image = batch[2]
                    plt.clf()
                    plt.imshow(ip.classes_to_rand_rgb(seg_image[0]).detach().permute(1, 2, 0).cpu().numpy())
                    plt.axis('off')
                    plt.savefig(f'{directory}/Train/seg_image.png', bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    torch.save(model.state_dict(), f'{directory}/Model/translator_{model_name}.pth')

        if (epoch + 1) % eval_every == 0:
            if rank == 0:
                eval(model, eval_dataloader, device)
                
    cleanup()
    
def run(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)
    
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2"
    n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run(main, world_size)
