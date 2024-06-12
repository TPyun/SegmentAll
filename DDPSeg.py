from cProfile import label
from hmac import new

from networkx import draw
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
import cv2
import sys
import ImageProcess as ip
import Model as md
import CocoDataLoader as ccdl
import Tele
from lineDetect import detect_lines

telegram = Tele.TeleSender()

    
def get_label_name(label_id):
    if label_id == 0:
        return 'Bac'
    elif label_id == 1:
        return 'Wall'
    elif label_id == 2:
        return 'Ped'
    elif label_id == 3:
        return 'Veh'
    else:
        return 'Invalid'
            
def channel_sum_loss(pred, loss_func):
    channel_sums = pred.sum(dim=1)  # 채널 방향으로 합산
    channel_sums = channel_sums.unsqueeze(1)
    target_sums = torch.ones_like(channel_sums)
    
    if loss_func == 'smooth':
        loss = F.smooth_l1_loss(channel_sums, target_sums, reduction='none').mean(dim=(2, 3))
    elif loss_func == 'mse':
        loss = F.mse_loss(channel_sums, target_sums, reduction='none').mean(dim=(2, 3))
    elif loss_func == 'iou':
        smooth = 1e-6
        intersection = (channel_sums * target_sums).float().sum(dim=(2, 3))
        union = (channel_sums + target_sums).float().sum(dim=(2, 3))
        iou = intersection / (union + smooth)
        loss = 1 - iou
        
    mean_loss = loss.mean(dim=1).mean()
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

def get_matching_channel_list(pred, target, loss_func='smooth'):
    num_pred = pred.size(0)
    num_target = target.size(0)
    target = (target > 0.5).float()
    target_sums = target.sum(dim=(1, 2))

    if loss_func == 'smooth':
        loss_fn = F.smooth_l1_loss
    elif loss_func == 'mse':
        loss_fn = F.mse_loss
    elif loss_func == 'bce':
        loss_fn = F.binary_cross_entropy

    loss_list = []
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

def rearrange_target(pred, target_seg, target_label):
    new_target = torch.zeros_like(target_seg)
    new_target_label = torch.zeros_like(target_label)

    # 배치 반복
    for b in range(pred.shape[0]):
        match_list = get_matching_channel_list(pred[b], target_seg[b])
        
        new_target_batch = torch.zeros_like(target_seg[b])
        new_target_label_batch = torch.zeros_like(target_label[b])
        
        for match in match_list:
            new_target_batch[match[0]] = target_seg[b][match[1]]
            new_target_label_batch[match[0]] = target_label[b][match[1]]
        
        new_target[b] = new_target_batch
        new_target_label[b] = new_target_label_batch

    return new_target, new_target_label

def get_label_info(label):
    label_info = []
    for c in range(label.shape[0]):
        max = 0
        max_index = 0
        for i in range(label.shape[1]):
            if label[c, i].item() >= max:
                max = label[c, i].item()
                max_index = i
        label_info.append(max_index)
    return label_info
    
def draw_instance_segmentation(image, segmentation, label, save_path):
    segmentation = segmentation > 0.5
    
    label_info = get_label_info(label)
    
    # print(label, label_info)
    
    plt.clf()
    plt.axis('off')
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.imshow(ip.classes_to_rand_rgb(segmentation).permute(1, 2, 0).cpu().numpy(), alpha=0.7)
    
    for channel in range(segmentation.shape[0]):
        if segmentation[channel].sum().item() == 0:
            continue
        # print(f"Channel: {channel}, Label: {label_info[channel]}")
        y, x = torch.where(segmentation[channel] == 1)
        mean_x = x.float().mean().item()
        mean_y = y.float().mean().item()
        this_label = label_info[channel]
        plt.text(mean_x, mean_y, f"{get_label_name(this_label)}", fontsize=7, color='black', ha='center', va='center')        
        
    plt.savefig(f'{directory}/{save_path}', bbox_inches='tight', pad_inches=0)
    plt.close()
    retult = cv2.imread(f'{directory}/{save_path}')
    return retult
    
def eval(model, eval_dataloader, device):
    global eval_plot_points, eval_iter
    
    model.eval()
    
    total_iou = 0
    total_except_error_iou = 0
    total_loss = 0
    total_inference_time = 0
    for batch_i, batch in enumerate(eval_dataloader):
        image = batch[0].to(device)
        seg_image = batch[1].to(device)
        target_label = batch[2].to(device)

        start_time = time.time()
        with torch.no_grad():
            result, pred_label = model(image)
        end_time = time.time()
        inference_time = end_time - start_time
        total_inference_time += inference_time
        
        result = result.detach()
        pred_label = pred_label.detach()
        
        seg_image, target_label = rearrange_target(result, seg_image, target_label)
        
        loss = F.smooth_l1_loss(result, seg_image)
        label_loss = F.smooth_l1_loss(pred_label, target_label)
        loss += label_loss
        result = result > 0.5
        
        iou, except_error_iou = get_iou(result, seg_image)

        total_loss += loss.item()
        total_iou += iou.item()
        total_except_error_iou += except_error_iou.item()
        
        progress = (batch_i + 1) / len(eval_dataloader)
        bar_length = 50
        bar = "#" * int(bar_length * progress) + "-" * (bar_length - int(bar_length * progress))
        sys.stdout.write(f'\r[{bar}] {progress * 100:.2f}%')
        sys.stdout.flush()
        
    total_loss /= len(eval_dataloader)
    total_iou /= len(eval_dataloader)
    total_except_error_iou /= len(eval_dataloader)
    total_inference_time /= len(eval_dataloader)
    
    print()
    print(f"Eval Loss: {total_loss:.8f}", end='| ')
    print(f"Eval IOU: {total_iou:.8f}", end='| ')
    print(f"Eval Except Error IOU: {total_except_error_iou:.8f}", end='| ')
    print(f"Eval Inference Time: {total_inference_time:.8f}")
    print()
    telegram.send_text(f"Eval Loss: {total_loss:.8f} | Eval IOU: {total_iou:.8f} | Eval Except Error IOU: {total_except_error_iou:.8f}")
    eval_plot_points.append([eval_iter, total_loss, total_iou, total_except_error_iou])   
    
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
    #         plt.imshow(ip.classes_to_rand_rgb(result[i]).detach().permute(1, 2, 0).cpu().numpy()) 
    #         plt.subplot(4, 3, 3 * i + 3)
    #         plt.subplots_adjust(wspace=0, hspace=0)
    #         plt.axis('off')
    #         plt.imshow(ip.classes_to_rand_rgb(seg_image[i]).detach().permute(1, 2, 0).cpu().numpy())
    #     except Exception as e:
    #         break
    # plt.savefig(f'{directory}/Eval/instance_{eval_iter}.png', bbox_inches='tight', pad_inches=0)
    # # plt.savefig(f'{directory}/Eval/instance{eval_iter}.png', bbox_inches='tight', pad_inches=0)
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
        plt.plot(point[0], point[2], 'bo')
    plt.savefig(f'{directory}/Eval/IOU.png', bbox_inches='tight', pad_inches=0)
    plt.close()    
    
    
    # # line detection
    # # result에서 선을 검출하여 선을 그린 이미지를 저장
    # line_image = np.zeros((result.shape[0], 1, result.shape[2], result.shape[3]), dtype=np.uint8)
    # np_result = result.cpu().numpy()
    
    
    # for b in range(result.shape[0]):
    #     label_info = get_label_info(pred_label[b])
    #     for c in range(result.shape[1]):
    #         if label_info[c] == 0 or label_info[c] == 1: 
    #             edges = cv2.Canny(np_result[b, c].astype(np.uint8), 0, 1)
    #             line_image[b, 0] = np.maximum(line_image[b, 0], edges)
                
    # detected_line_image = detect_lines(line_image[0, 0])
    
    # pred_result = draw_instance_segmentation(image[0], result[0], pred_label[0], f'Eval/instance_{eval_iter}.png')
    # plt.clf()
    # plt.figure(figsize=(9, 3))
    # plt.subplot(1, 3, 1)
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.axis('off')
    # plt.imshow(np.clip(image[0].permute(1, 2, 0).cpu().numpy(), 0, 1))
    # plt.subplot(1, 3, 2)
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.axis('off')
    # plt.imshow(pred_result)
    # plt.subplot(1, 3, 3)
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(detected_line_image, cv2.COLOR_BGR2RGB))
    # plt.savefig(f'{directory}/Eval/lineDetect_{eval_iter}.png', bbox_inches='tight', pad_inches=0)
    # plt.close()
    
            
    pred_result = draw_instance_segmentation(image[0], result[0], pred_label[0], f'Eval/instance_{eval_iter}.png')
    gt_result = draw_instance_segmentation(image[0], seg_image[0], target_label[0], f'Eval/instance_real_{eval_iter}.png')
    plt.clf()
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(pred_result)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(gt_result)
    plt.savefig(f'{directory}/Eval/instance_total.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    model.train()
    eval_iter += 1     

script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)
os.chdir(directory)

width = 256
height = 256
actual_batch_size = 2
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
    
    model = md.SimpleSegmentationModel().to(device)
    model_name = model.__class__.__name__
    model_path = f'{directory}/Model/translator_{model_name}.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=map_location))
        print(f"Loaded {model_path}!")
    else:   
        print(f"Model {model_path} not exists.")
    ddp_model = DDP(model, device_ids=[rank])
    
    optimizer = optim.NAdam(ddp_model.parameters(), lr=0.0001)

    root = f'{directory}/COCO'
    train_dataset = ccdl.PanopticCocoDataset(root, 'train2017', width, height)
    
    # root = f'{directory}/SAMDataset'
    # train_dataset = ccdl.SAMDataset(root, 'train', width, height)
    
    # root = '/home/wooyung/Develop/RadarDetection/20240404/'
    # train_dataset = ccdl.NewSegDataset(root, width, height, mode='train', random=True)
    
    # root = '/home/wooyung/Develop/RadarDetection/CocoSeg/Mapillary/'
    # train_dataset = ccdl.MapillaryData(root, width, height)
    
    # root = '/home/wooyung/Develop/RadarDetection/SegmentAll/MapillaryPanopticSet_256/'
    # train_dataset = ccdl.PreProcessedMapillaryDataset(root, width, height, type='train')

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=actual_batch_size, sampler=train_sampler, pin_memory=True, num_workers=4)
    
    if rank == 0:
        eval_dataset = ccdl.PanopticCocoDataset(root, 'val2017', width, height)
        # eval_dataset = ccdl.PreProcessedMapillaryDataset(root, width, height, type='eval')
        # eval_dataset = ccdl.MapillaryData(root, width, height, 'eval')
        # root = '/home/wooyung/Develop/RadarDetection/Image_Label_5fps/'
        # eval_dataset = ccdl.NewSegDataset(root, width, height, mode='train', random=False)
        # eval_dataset = ccdl.SAMDataset(root, 'val', width, height)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=True)
        
    
    num_epochs = 100000
    print_every = 1
    eval_every = 10
    train = True
    
    t_loss_epoch = 0
    t_iou_epoch = 0
    t_iou_except_error_epoch = 0
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    
    for epoch in range(num_epochs):
        
        if train:
            train_sampler.set_epoch(epoch)
            
            loss_epoch = 0
            iou_epoch = 0
            iou_except_error_epoch = 0
            epoch_start_time = time.time()
            
            for batch_i, batch in enumerate(train_dataloader):
                image = batch[0].to(device)
                seg_image = batch[1].to(device)
                target_label = batch[2].to(device)

                # 객체 탐지 모델 학습
                result, pred_label = ddp_model(image)
                seg_image, target_label = rearrange_target(result, seg_image, target_label)
                    
                loss = F.smooth_l1_loss(result, seg_image)
                label_loss = F.smooth_l1_loss(pred_label, target_label)
                loss += label_loss
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                if rank == 0:
                    result = result.detach()
                    # result = ip.make_max_image(result, limit=0.5)
                    result = result > 0.5
                    iou, except_error_iou = get_iou(result, seg_image)
                    
                    loss_epoch += loss.item()
                    iou_epoch += iou.item()
                    iou_except_error_epoch += except_error_iou.item()
                
                    progress = (batch_i + 1) / len(train_dataloader)
                    bar_length = 50
                    bar = "#" * int(bar_length * progress) + "-" * (bar_length - int(bar_length * progress))
                    
                    elapsed_time = time.time() - epoch_start_time
                    epoch_time_estimate = elapsed_time / (batch_i + 1) * len(train_dataloader)
                    formatted_time = time.strftime("%H:%M:%S", time.gmtime(epoch_time_estimate))
                    
                    sys.stdout.write(f'\r[{bar}] {progress * 100:.2f}% Loss:{loss.item():.4f} IOU:{iou.item():.4f} EEIOU:{except_error_iou.item():.4f} ET:{formatted_time}')
                    sys.stdout.flush()
                    
            loss_epoch /= len(train_dataloader)
            t_loss_epoch += loss_epoch
            iou_epoch /= len(train_dataloader)
            t_iou_epoch += iou_epoch
            iou_except_error_epoch /= len(train_dataloader)
            t_iou_except_error_epoch += iou_except_error_epoch
            
            if (epoch + 1) % print_every == 0:
                if rank == 0:
                    print()
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", end='| ')
                    print(f"epoch: {epoch}", end='| ')
                    print(f"loss: {t_loss_epoch / print_every:.8f}", end='| ')
                    print(f"IOU: {t_iou_epoch / print_every:.8f}", end='| ')
                    print(f"Except Error IOU: {t_iou_except_error_epoch / print_every:.8f}", end='| ')
                    # print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
                    print()
                    telegram.send_text(f"epoch: {epoch}, loss: {t_loss_epoch / print_every:.8f}, IOU: {t_iou_epoch / print_every:.8f}, Except Error IOU: {t_iou_except_error_epoch / print_every:.8f}")
                    train_plot_points.append([epoch, t_loss_epoch / print_every, t_iou_epoch / print_every, t_iou_except_error_epoch / print_every])
                    
                    
                t_loss_epoch = 0
                t_iou_epoch = 0
                t_iou_except_error_epoch = 0
                
                if rank == 0:
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
                    #         plt.imshow(ip.classes_to_rand_rgb(result[i]).permute(1, 2, 0).cpu().numpy()) 
                    #         plt.subplot(4, 3, 3 * i + 3)
                    #         plt.subplots_adjust(wspace=0, hspace=0)
                    #         plt.axis('off')
                    #         plt.imshow(ip.classes_to_rand_rgb(seg_image[i]).permute(1, 2, 0).cpu().numpy())
                    #     except Exception as e:
                    #         break
                    # plt.savefig(f'{directory}/Train/instance.png', bbox_inches='tight', pad_inches=0)
                    # plt.close()
                    
                    # seg_image 각 채널 시각화
                    plt.clf()
                    plt.figure(figsize=(12, 12))
                    for i in range(32):
                        plt.subplot(8, 8, i + 1)
                        plt.subplots_adjust(wspace=0.1, hspace=0.1)
                        plt.axis('off')
                        plt.imshow(result[0, i].detach().cpu().numpy())
                    for i in range(32):
                        plt.subplot(8, 8, i + 1 + 32)
                        plt.subplots_adjust(wspace=0.1, hspace=0.1)
                        plt.axis('off')
                        plt.imshow(seg_image[0, i].detach().cpu().numpy())
                    plt.savefig(f'{directory}/Train/seg_image.png', bbox_inches='tight', pad_inches=0)
                    plt.close()
                
                    draw_instance_segmentation(image[0], result[0], pred_label[0], f'Train/instance.png')
                    draw_instance_segmentation(image[0], seg_image[0], target_label[0], f'Train/instance_real.png')
                    
                    plt.clf()
                    plt.figure(figsize=(12, 12))
                    for point in train_plot_points:
                        plt.plot(point[0], point[1], 'ro')
                    plt.savefig(f'{directory}/Train/loss.png', bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    plt.clf()
                    plt.figure(figsize=(12, 12))
                    for point in train_plot_points:
                        plt.plot(point[0], point[2], 'bo')
                    plt.savefig(f'{directory}/Train/IOU.png', bbox_inches='tight', pad_inches=0)
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
    os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"
    n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run(main, world_size)




