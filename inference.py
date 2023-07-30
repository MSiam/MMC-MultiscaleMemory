"""
Based on VisTR (https://github.com/Epiphqny/VisTR)
and DETR (https://github.com/facebookresearch/detr)
Usage:
    MoCA: python inference.py --model_path <your_model_path.pth> --output_dir <your_output_dir>  --dataset moca --val_size 473 --aug
    Youtube Objects: python inference.py --model_path <your_model_path.pth> --output_dir <your_output_dir>  --dataset ytbo --val_size 360 --aug
    Davis16: python inference.py --model_path <your_model_path.pth> --output_dir <your_output_dir>  --dataset davis --val_size 473 --aug
"""
import argparse
import logging
import random
import numpy as np
import cv2
import ast
import operator
import csv
import pandas as pd
from tqdm import tqdm
import os
import glob
import torch
import time
from torch.utils.data import DataLoader
from src.datasets import path_config as dataset_path_config
from src.datasets.test.davis16_val_data import Davis16ValDataset as Davis16ValDataset
from src.datasets.test.moca import MoCADataset
from src.datasets.test.youtube_objects import YouTubeObjects
import src.util.misc as utils
from src.util import misc
from src.models.vos.med_vt_swin import build_model_swin_medvt as build_model
import nvidia_smi

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def compute_gpu_memory(cuda_device):
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(cuda_device)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    nvidia_smi.nvmlShutdown()
    return info.used / 1000000

def get_args_parser():
    parser = argparse.ArgumentParser('MED-VT', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    # Model parameters
    parser.add_argument('--model_path', type=str, default=None, required=True,
                        help="Path to the model weights.")
    parser.add_argument('--finetune', action='store_true', default=False)

    # * Init Weights
    parser.add_argument('--resnet50_coco_weights_path', type=str,
                        default="/local/riemann/home/rezaul/projects/medvt2-main/pretrained/detr_coco/384_coco_r50.pth",
                        help="Path to the pretrained model.")
    parser.add_argument('--resnet101_coco_weights_path', type=str,
                        default="/local/riemann/home/rezaul/projects/medvt2-main/pretrained/detr_coco/384_coco_r101.pth",
                        help="Path to the pretrained model.")
    parser.add_argument('--swin_s_pretrained_path', type=str,
                        default="/local/riemann/home/rezaul/projects/medvt2-main/pretrained/swin_small_patch244_window877_kinetics400_1k.pth",
                        help="swin-s pretrained model path.")
    parser.add_argument('--swin_b_pretrained_path', type=str,
                        default="/local/riemann/home/rezaul/projects/medvt2-main/pretrained/swin_base_patch244_window877_kinetics400_22k.pth",
                        help="swin-s pretrained model path.")
    parser.add_argument('--swin_b_pretrained_path_k600', type=str,
                        default="/local/riemann/home/rezaul/projects/medvt2-main/pretrained/swin_base_patch244_window877_kinetics600_22k.pth",
                        help="swin-s pretrained model path.")
    parser.add_argument('--lr_backbone', default=1e-6, type=float)

    # * Backbone
    parser.add_argument('--backbone', default='swinB', type=str,
                        help="backbone to use, [resnet50, resnet101, swinS, swinB]")
    parser.add_argument('--model_type', default='encoder-decoder', type=str,
                        help="backbone to use, [encoder-decoder, vanilla-fpn, vanilla]")

    parser.add_argument('--use_fpn', default='True', type=bool, help="to use fpn or not")
    parser.add_argument('--dilation', default=False, action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=(6,), type=tuple,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--enc_separable_attn', default='thw', type=str,
                        help="Encoder separable attn , (thw, t-hw, th-tw, th-tw-hw )")
    parser.add_argument('--encoder_cross_layer', default=False, type=bool,
                        help="Cross resolution attention")

    parser.add_argument('--dec_layers', default=9, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--decoder_type', type=str, default='multiscale_query')
    parser.add_argument('--num_frames', default=5, type=int,
                        help="Number of frames")
    parser.add_argument('--num_queries', default=5, type=int,
                        help="Number of query slots")
    parser.add_argument('--val_size', default=473, type=int,
                        help="input size of image for evaluation")
    parser.add_argument('--temporal_strides', default=[1], nargs='+', type=int,
                        help="temporal strides used to construct input cip")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--lprop_mode', default=0, type=int, help='Choose 1 for unidir lprop, 2 for bidir lprop, 3 bidir no learning, 0 for no lprop')
    parser.add_argument('--feat_loc', default='late', type=str, help='Choose early or late features to compare in lprop')
    parser.add_argument('--pretrain_settings', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--use_soft_mask_encoder', default=False, type=bool,
                        help="use soft mask")
    parser.add_argument('--stacked_lprop', type=int, default=1, help="repeat the lprop")

    parser.add_argument('--use_mem_mask', action="store_true", default=False,
                        help="history mask from previous clip")
    parser.add_argument('--mask_consistency_decoder', default=False, type=bool,
                        help="discarded, too much memory")

    # LOSS
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument("--save_pred", action="store_true", default=False)
    # * Segmentation
    parser.add_argument('--masks', action='store_true', default=True,
                        help="Train segmentation head if the flag is provided")

    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--output_dir', required=True,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--compute_memory', action='store_true')
    return parser


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_predictions_flip_ms(model, samples, targets, gt_shape, ms=True, ms_gather='mean', flip=True,
                                flip_gather='mean', scales=None, clip_tag=None):
    clip_tag_flipped = None
    if clip_tag is not None:
        clip_tag_flipped = '%s@flipped' % clip_tag

    outputs = compute_predictions_ms(model, samples, targets, gt_shape, ms=ms, ms_gather=ms_gather,
                                     scales=scales, clip_tag=clip_tag)

    outputs['pred_masks'] = utils.interpolate(outputs['pred_masks'], size=gt_shape, mode="bilinear",
                                              align_corners=False)
    if flip:
        samples_flipped, targets_flipped = augment_flip(samples, targets)
        outputs_flipped = compute_predictions_ms(model, samples_flipped, targets_flipped, gt_shape, ms=ms,
                                                 ms_gather=ms_gather, scales=scales, clip_tag=clip_tag_flipped)
        outputs_flipped['pred_masks'] = utils.interpolate(outputs_flipped['pred_masks'], size=gt_shape,
                                                          mode="bilinear", align_corners=False)
        if flip_gather == 'max':
            outputs['pred_masks'] = torch.max(outputs_flipped['pred_masks'].flip(-1), outputs['pred_masks'])
        else:
            outputs['pred_masks'] = (outputs_flipped['pred_masks'].flip(-1) + outputs['pred_masks']) / 2.0
    return outputs


def compute_predictions_ms(model, samples, targets, gt_shape, ms=True, ms_gather='mean', scales=None, clip_tag=None):
    if ms:
        if scales is None:
            scales = [0.7, 0.8, 0.9, 1, 1.1, 1.2]
    else:
        scales = [1]
    mask_list = []
    org_shape = samples.tensors.shape[-2:]
    for scale in scales:
        size = [int(val * scale) for val in org_shape]
        tensors = utils.interpolate(samples.tensors, size=size, mode="bilinear", align_corners=False)
        mask = utils.interpolate(samples.mask.unsqueeze(1).long().float(), size=size, mode="nearest").squeeze(1)
        mask[mask > 0.5] = True
        mask[mask <= 0.5] = False
        mask = mask.bool()
        ms_sample = misc.NestedTensor(tensors, mask)
        model_output = model(ms_sample, clip_tag=clip_tag)
        pred = utils.interpolate(model_output['pred_masks'], size=gt_shape, mode="bilinear", align_corners=False)
        pred = pred.sigmoid()
        mask_list.append(pred)
    if ms:
        if ms_gather == 'max':
            ms_pred = torch.max(torch.stack(mask_list, dim=0), dim=0)
            output_result = {'pred_masks': ms_pred}
        else:
            ms_pred = torch.mean(torch.stack(mask_list, dim=0), dim=0)
            output_result = {'pred_masks': ms_pred}
    else:
        output_result = {'pred_masks': mask_list[0]}
    return output_result


def augment_flip(samples, targets, dim=-1):
    samples.tensors = samples.tensors.flip(dim)
    samples.mask = samples.mask.flip(dim)
    return samples, targets


def computeWeightedProbs(SalMap2D):
    weights = np.linspace(0.1, 0.9, 10)
    counter = -1
    iSalMap = np.zeros_like(SalMap2D)
    for th in weights:
        counter = counter + 1
        BinTh = SalMap2D.copy()
        BinTh[BinTh <= th] = 0.0
        BinTh[BinTh > th] = 1.0
        iSalMap = iSalMap + th * BinTh
    iSalMap = iSalMap / np.sum(weights)
    return iSalMap


def moca_read_annotation(annotation):
    reader = csv.reader(open(annotation, 'r'))
    next(reader, None)
    d = {}
    reader = sorted(reader, key=operator.itemgetter(1))
    for row in reader:
        _, fn, bbox, motion = row
        if bbox != '[]':
            if motion == '{}':
                motion = old_motion
            old_motion = motion
            name = fn.split('/')[-2]
            number = fn.split('/')[-1][:-4]
            if name not in d:
                d[name] = {}
            if number not in d[name]:
                d[name][number] = {}
            d[name][number]['fn'] = fn
            motion = ast.literal_eval(motion)
            d[name][number]['motion'] = motion['1']
            bbox = ast.literal_eval(bbox)
            _, xmin, ymin, width, height = list(bbox)
            xmin = max(xmin, 0.)
            ymin = max(ymin, 0.)
            d[name][number]['xmin'] = xmin
            d[name][number]['xmax'] = xmin + width
            d[name][number]['ymin'] = ymin
            d[name][number]['ymax'] = ymin + height
    return d


def moca_bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def moca_heuristic_fg_bg(mask):
    mask = mask.copy()
    h, w = mask.shape
    mask[1:-1, 1:-1] = 0
    borders = 2 * h + 2 * w - 4
    return np.sum(mask > 127.5) / borders


def moca_eval(out_dir='./results/moca', resize=1):
    moca_dataset_path = dataset_path_config.moca_dataset_images_path
    moca_csv = dataset_path_config.moca_annotations_csv
    moca_pred_dir = os.path.join(out_dir)
    if not os.path.exists(moca_pred_dir):
        raise ValueError('MoCA Pred Dir not exists!!!')

    out_csv = os.path.join(out_dir, 'MoCA_results.csv')

    with open(out_csv, 'w') as f:
        df = pd.DataFrame([], columns=['Seq_name', 'Locomotion_IoU', 'Locomotion_S_0.5', 'Locomotion_S_0.6',
                                       'Locomotion_S_0.7', 'Locomotion_S_0.8', 'Locomotion_S_0.9',
                                       'Deformation_IoU', 'Deformation_S_0.5', 'Deformation_S_0.6', 'Deformation_S_0.7',
                                       'Deformation_S_0.8', 'Deformation_S_0.9',
                                       'Static_IoU', 'Static_S_0.5', 'Static_S_0.6', 'Static_S_0.7', 'Static_S_0.8',
                                       'Static_S_0.9',
                                       'All_motion_IoU', 'All_motion_S_0.5', 'All_motion_S_0.6', 'All_motion_S_0.7',
                                       'All_motion_S_0.8', 'All_motion_S_0.9',
                                       'locomotion_num', 'deformation_num', 'static_num'])

        df.to_csv(f, index=False,
                  columns=['Seq_name', 'Locomotion_IoU', 'Locomotion_S_0.5', 'Locomotion_S_0.6', 'Locomotion_S_0.7',
                           'Locomotion_S_0.8', 'Locomotion_S_0.9',
                           'Deformation_IoU', 'Deformation_S_0.5', 'Deformation_S_0.6', 'Deformation_S_0.7',
                           'Deformation_S_0.8', 'Deformation_S_0.9',
                           'Static_IoU', 'Static_S_0.5', 'Static_S_0.6', 'Static_S_0.7', 'Static_S_0.8', 'Static_S_0.9',
                           'All_motion_IoU', 'All_motion_S_0.5', 'All_motion_S_0.6', 'All_motion_S_0.7',
                           'All_motion_S_0.8', 'All_motion_S_0.9',
                           'locomotion_num', 'deformation_num', 'static_num'])
        pass

    annotations = moca_read_annotation(moca_csv)
    Js = AverageMeter()

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    success_rates_overall = np.zeros(5)
    total_frames_l = 0
    total_frames_d = 0
    total_frames_s = 0
    success_l_overall = [0, 0, 0, 0, 0]
    success_d_overall = [0, 0, 0, 0, 0]
    success_s_overall = [0, 0, 0, 0, 0]

    video_names = sorted(os.listdir(moca_pred_dir))
    for video in video_names:
        if video not in annotations:
            continue
        res_path = os.path.join(moca_pred_dir, video)
        res_list = sorted([f for f in glob.glob(res_path + '/*.png', recursive=False)])  # for our model

        n_frames = len(res_list)
        js = []

        iou_static = AverageMeter()
        iou_locomotion = AverageMeter()
        iou_deformation = AverageMeter()
        ns = 0
        nl = 0
        nd = 0
        success_l = [0, 0, 0, 0, 0]
        success_d = [0, 0, 0, 0, 0]
        success_s = [0, 0, 0, 0, 0]
        if resize:
            image = np.array(cv2.imread(os.path.join(moca_dataset_path, video, '{:05d}.jpg'.format(0))))
            H, W, _ = image.shape
        for ff in range(n_frames):
            number = str(ff).zfill(5)
            if number in annotations[video]:
                # get annotation
                motion = annotations[video][number]['motion']
                x_min = annotations[video][number]['xmin']
                x_max = annotations[video][number]['xmax']
                y_min = annotations[video][number]['ymin']
                y_max = annotations[video][number]['ymax']
                bbox_gt = [x_min, y_min, x_max, y_max]

                # get mask
                mask = np.array(cv2.imread(res_list[ff]), dtype=np.uint8)
                if len(mask.shape) > 2:
                    mask = mask.mean(2)
                H_, W_ = mask.shape

                if moca_heuristic_fg_bg(mask) > 0.5:
                    mask = (255 - mask).astype(np.uint8)

                thres = 0.1 * 255
                mask[mask > thres] = 255
                mask[mask <= thres] = 0

                contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                area = 0

                for cnt in contours:
                    (x_, y_, w_, h_) = cv2.boundingRect(cnt)
                    area_ = np.sum(mask[y_:y_ + h_, x_:x_ + w_])
                    if area_ > area:
                        x = x_
                        y = y_
                        w = w_
                        h = h_
                        area = area_
                H_, W_ = mask.shape
                if area > 0:
                    bbox = np.array([x, y, x + w, y + h], dtype=float)
                    # if the size reference for the annotation (the original jpg image) is different from the size of the mask
                    if resize:
                        bbox[0] *= W / W_
                        bbox[2] *= W / W_
                        bbox[1] *= H / H_
                        bbox[3] *= H / H_
                    iou = moca_bb_intersection_over_union(np.array(bbox_gt, dtype=float), np.array(bbox, dtype=float))
                else:
                    iou = 0.
                js.append(iou)

                # get motion
                if motion == '1':
                    iou_deformation.update(iou)
                    nd += 1
                    for k in range(len(thresholds)):
                        success_d[k] += int(iou > thresholds[k])

                elif motion == '0':
                    iou_locomotion.update(iou)
                    nl += 1
                    for k in range(len(thresholds)):
                        success_l[k] += int(iou > thresholds[k])

                elif motion == '2':
                    iou_static.update(iou)
                    ns += 1
                    for k in range(len(thresholds)):
                        success_s[k] += int(iou > thresholds[k])

        total_frames_l += nl
        total_frames_s += ns
        total_frames_d += nd
        for k in range(len(thresholds)):
            success_l_overall[k] += success_l[k]
            success_s_overall[k] += success_s[k]
            success_d_overall[k] += success_d[k]

        js_m = np.average(js)
        locomotion_val = -1.
        static_val = -1.
        deformation_val = -1.
        if iou_locomotion.count > 0:
            locomotion_val = iou_locomotion.avg
        if iou_deformation.count > 0:
            deformation_val = iou_deformation.avg
        if iou_static.count > 0:
            static_val = iou_static.avg

        all_motion_S = np.array(success_l) + np.array(success_s) + np.array(success_d)
        success_rates_overall += all_motion_S
        with open(out_csv, 'a') as f:
            raw_data = {'Seq_name': video, 'Locomotion_IoU': [locomotion_val],
                        'Locomotion_S_0.5': [success_l[0]], 'Locomotion_S_0.6': [success_l[1]],
                        'Locomotion_S_0.7': [success_l[2]], 'Locomotion_S_0.8': [success_l[3]],
                        'Locomotion_S_0.9': [success_l[4]],
                        'Deformation_IoU': [deformation_val],
                        'Deformation_S_0.5': [success_d[0]], 'Deformation_S_0.6': [success_d[1]],
                        'Deformation_S_0.7': [success_d[2]], 'Deformation_S_0.8': [success_d[3]],
                        'Deformation_S_0.9': [success_d[4]],
                        'Static_IoU': [static_val],
                        'Static_S_0.5': [success_s[0]], 'Static_S_0.6': [success_s[1]],
                        'Static_S_0.7': [success_s[2]], 'Static_S_0.8': [success_s[3]],
                        'Static_S_0.9': [success_s[4]],
                        'All_motion_IoU': [js_m],
                        'All_motion_S_0.5': [all_motion_S[0]], 'All_motion_S_0.6': [all_motion_S[1]],
                        'All_motion_S_0.7': [all_motion_S[2]], 'All_motion_S_0.8': [all_motion_S[3]],
                        'All_motion_S_0.9': [all_motion_S[4]],
                        'locomotion_num': [nl], 'deformation_num': [nd], 'static_num': [ns]}
            df = pd.DataFrame(raw_data, columns=['Seq_name', 'Locomotion_IoU', 'Locomotion_S_0.5', 'Locomotion_S_0.6',
                                                 'Locomotion_S_0.7', 'Locomotion_S_0.8', 'Locomotion_S_0.9',
                                                 'Deformation_IoU', 'Deformation_S_0.5', 'Deformation_S_0.6',
                                                 'Deformation_S_0.7', 'Deformation_S_0.8', 'Deformation_S_0.9',
                                                 'Static_IoU', 'Static_S_0.5', 'Static_S_0.6', 'Static_S_0.7',
                                                 'Static_S_0.8', 'Static_S_0.9',
                                                 'All_motion_IoU', 'All_motion_S_0.5', 'All_motion_S_0.6',
                                                 'All_motion_S_0.7', 'All_motion_S_0.8', 'All_motion_S_0.9',
                                                 'locomotion_num', 'deformation_num', 'static_num'])
            df.to_csv(f, header=False, index=False)
        Js.update(js_m)

    success_rates_overall = success_rates_overall / (total_frames_l + total_frames_s + total_frames_d)
    info = 'Overall :  Js: ({:.3f}). SR at '
    for k in range(len(thresholds)):
        info += str(thresholds[k])
        info += ': ({:.3f}), '
    info = info.format(Js.avg, success_rates_overall[0], success_rates_overall[1], success_rates_overall[2],
                       success_rates_overall[3], success_rates_overall[4])
    print('dataset: MoCA result:%s' % info)
    print('dataset: MoCA mean_iou:%0.3f' % Js.avg)
    logger.debug('dataset: MoCA result:%s' % info)
    logger.debug('dataset: MoCA mean_iou:%0.3f' % Js.avg)
    return Js.avg, info


def moca_infer(model, data_loader, device, aug=False, save_pred=False, out_dir='./results/moca/', lprop_mode=0, compute_memory=False):
    model.eval()
    i_iter = 0
    #niters_to_compute_mem = 100
    moca_csv = dataset_path_config.moca_annotations_csv
    annotations = moca_read_annotation(moca_csv)
    for samples, targets in tqdm(data_loader):
        i_iter = i_iter + 1
        if compute_memory:
            if i_iter == 1:
                avg_mem = 0

            #if i_iter > niters_to_compute_mem:
            #    break

        video_name = targets[0]['video_name']
        center_frame = targets[0]['center_frame']
        if video_name not in annotations:
            continue
        if center_frame not in annotations[video_name]:
            continue
        frame_ids = targets[0]['frame_ids']
        center_frame_index = frame_ids.index(center_frame)
        samples = samples.to(device)
        targets = [{k: v.to(device) if k in ['masks'] else v for k, v in t.items()} for t in targets]
        gt_shape = samples.tensors.shape[-2:]
        if aug:
            outputs = compute_predictions_flip_ms(model, samples, targets, gt_shape, ms=True, ms_gather='mean',
                                                  flip=True, flip_gather='max',
                                                  scales=[0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                                  clip_tag=video_name)
        else:
            outputs = model(samples)
            #end_mem = torch.cuda.memory_allocated()

            outputs['pred_masks'] = utils.interpolate(outputs['pred_masks'], size=gt_shape, mode="bilinear",
                                                      align_corners=False)
            if lprop_mode != 3:
                outputs['pred_masks'] = outputs['pred_masks'].sigmoid()
            samples_flipped, targets_flipped = augment_flip(samples, targets)
            outputs_flipped = model(samples_flipped, clip_tag='%s@flipped'%video_name)
            outputs_flipped['pred_masks'] = utils.interpolate(outputs_flipped['pred_masks'], size=gt_shape,
                                                              mode="bilinear", align_corners=False)
            if lprop_mode != 3:
                outputs_flipped['pred_masks'] = outputs_flipped['pred_masks'].sigmoid()
            outputs['pred_masks'] = (outputs_flipped['pred_masks'].flip(-1) + outputs['pred_masks']) / 2.0

            if compute_memory:
                end_mem = compute_gpu_memory(1)
                avg_mem += end_mem

        src_masks = outputs["pred_masks"]
        yc = src_masks[0].cpu().detach().numpy().copy()
        mask = yc[center_frame_index, :, :]
        mask[mask > 0.1] = 255
        mask[mask <= 0.1] = 0
        mask = mask.astype(np.uint8)
        if save_pred:
            pred_out_dir = os.path.join(out_dir, video_name)
            if not os.path.exists(pred_out_dir):
                os.makedirs(pred_out_dir)
            cv2.imwrite(os.path.join(pred_out_dir, '%s.png' % center_frame), mask)

    if compute_memory:
        niters_to_compute_mem = i_iter
        print('Average memory usage is ', avg_mem/niters_to_compute_mem)
    return


def infer_ytbobj_perseqpercls(model, data_loader, device, aug=False, save_pred=False,
                              out_dir='./results/youtube_objects/'):
    model.eval()
    i_iter = 0
    percls_perseq_iou_dict = {}
    num_iou_dict = {}
    total_mask = 0
    running_video_name = None
    for samples, targets in tqdm(data_loader):
        i_iter = i_iter + 1
        # import ipdb; ipdb.set_trace()
        video_name = targets[0]['video_name']
        video_name = video_name.split('/')[1]
        frame_ids = targets[0]['frame_ids']
        center_frame_name = targets[0]['center_frame']
        center_frame_index = frame_ids.index(center_frame_name)
        center_img_path = targets[0]['img_paths'][center_frame_index]
        img_dir, frame_name = os.path.split(center_img_path)
        mask_dir = img_dir.replace('frames', 'youtube_masks')
        frame_no = frame_name.replace('frame', '').replace('.jpg', '')
        mask_frame_name = '%05d.jpg' % int(frame_no)
        mask_file = os.path.join(mask_dir, 'labels', mask_frame_name)
        samples = samples.to(device)
        targets = [{k: v.to(device) if k in ['masks'] else v for k, v in t.items()} for t in targets]
        if running_video_name is not None and video_name != running_video_name:
            cls_mean = np.mean([seq_sum / count for seq_sum, count in
                                zip(percls_perseq_iou_dict[running_video_name].values(),
                                    num_iou_dict[running_video_name].values())])
            print('class_name:%s  iou:%0.3f' % (running_video_name, cls_mean))
        running_video_name = video_name
        tokens2 = mask_file.split('/')
        object_ = tokens2[-7]
        seq_name = tokens2[-5]
        clip_tag = f'{object_}.{seq_name}'
        if object_ not in percls_perseq_iou_dict:
            percls_perseq_iou_dict[object_] = {}
            num_iou_dict[object_] = {}
        if seq_name not in percls_perseq_iou_dict[object_]:
            percls_perseq_iou_dict[object_][seq_name] = 0
            num_iou_dict[object_][seq_name] = 0
        if not os.path.exists(mask_file):
            continue
        total_mask += 1
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        gt_shape = mask.shape
        th_mask = 80
        mask[mask < th_mask] = 0
        mask[mask >= th_mask] = 1
        th_pred = 110 / 255.0
        if aug:
            outputs = compute_predictions_flip_ms(model, samples, targets, gt_shape, ms=True, ms_gather='mean',
                                                  flip=True, flip_gather='max', scales=[0.95, 1, 1.05],
                                                  clip_tag=clip_tag)
        else:
            outputs = model(samples, clip_tag=clip_tag)
            outputs['pred_masks'] = utils.interpolate(outputs['pred_masks'], size=gt_shape, mode="bilinear",
                                                      align_corners=False)
            outputs['pred_masks'] = outputs['pred_masks'].sigmoid()
            samples_flipped, targets_flipped = augment_flip(samples, targets)
            outputs_flipped = model(samples_flipped, clip_tag='%s@flipped'%clip_tag)
            outputs_flipped['pred_masks'] = utils.interpolate(outputs_flipped['pred_masks'], size=gt_shape,
                                                              mode="bilinear", align_corners=False)
            outputs_flipped['pred_masks'] = outputs_flipped['pred_masks'].sigmoid()
            outputs['pred_masks'] = torch.max(outputs_flipped['pred_masks'].flip(-1), outputs['pred_masks'])
        src_masks = outputs["pred_masks"]
        yc_logits = src_masks[0].cpu().detach().numpy()[center_frame_index, :, :].copy()
        bin_mask = yc_logits.copy()
        bin_mask[bin_mask < th_pred] = 0
        bin_mask[bin_mask >= th_pred] = 1
        out = bin_mask.astype(mask.dtype)
        # ###########################
        iou = db_eval_iou(mask.copy(), out.copy())
        percls_perseq_iou_dict[object_][seq_name] += iou
        num_iou_dict[object_][seq_name] += 1
        if save_pred:
            pred_out_dir = os.path.join(out_dir, '/'.join(
                mask_dir.split('/')[-5:]))  # this one follows the same dir struck as gt
            if not os.path.exists(pred_out_dir):
                os.makedirs(pred_out_dir)
            cv2.imwrite(os.path.join(pred_out_dir, '%s.png' % center_frame_name),
                        out.astype(np.float) * 255)  # it was 0, 1
    percls_perseq = np.mean([seq_sum / count for seq_sum, count in
                             zip(percls_perseq_iou_dict[running_video_name].values(),
                                 num_iou_dict[running_video_name].values())])
    print('class_name:%s iou:%0.3f' % (running_video_name, percls_perseq))
    print('total_masks:%d' % total_mask)
    # ### ### Write the results to CSV ### ###
    print('****************************************************************')
    print('***************Youtube-Objects Eval Results**********************')
    logger.debug('****************************************************************')
    logger.debug('***************Youtube-Objects Eval Results**********************')
    iou_objs = {}
    for obj in percls_perseq_iou_dict.keys():
        for seq in percls_perseq_iou_dict[obj].keys():
            percls_perseq_iou_dict[obj][seq] /= num_iou_dict[obj][seq]
        iou_objs[obj] = np.mean(list(percls_perseq_iou_dict[obj].values()))
    overall_iou = np.mean(list(iou_objs.values()))
    for obj, obj_iou in iou_objs.items():
        print('IoU reported for  %s is %0.3f' % (obj, obj_iou))
        logger.debug('IoU reported for  %s is %0.3f' % (obj, obj_iou))
    print('Average IoU : %0.3f' % overall_iou)
    logger.debug('Average IoU : %0.3f' % overall_iou)
    logger.debug('****************************************************************')
    print('****************************************************************')
    # write_youtubeobjects_results_to_csv(out_dir, 'youtube_objects_results.csv', iou_objs)
    # write results to csv
    csv_file_name = 'youtube_objects_results.csv'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    obj_names = []
    obj_iou = []
    for k, v in iou_objs.items():
        obj_names.append(k)
        obj_iou.append('%0.3f' % v)
    obj_names.append('Overall')
    obj_iou.append('%0.3f' % np.mean(list(iou_objs.values())))
    with open(os.path.join(out_dir, csv_file_name), 'w') as f:
        cf = csv.writer(f)
        cf.writerow(obj_names)
        cf.writerow(obj_iou)
    return


def db_eval_iou(annotation, segmentation):
    """
    Collected from https://github.com/fperazzi/davis/blob/main/python/lib/davis/measures/jaccard.py
    Compute region similarity as the Jaccard Index.
         Arguments:
             annotation   (ndarray): binary annotation   map.
             segmentation (ndarray): binary segmentation map.
         Return:
             jaccard (float): region similarity
    """
    annotation = annotation.astype(np.bool_)
    segmentation = segmentation.astype(np.bool_)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
               np.sum((annotation | segmentation), dtype=np.float32)


def infer_on_davis(model, data_loader, device, save_pred=False, aug=False, out_dir='./results/davis/'):
    model.eval()
    i_iter = 0
    iou_list = []
    vid_iou_dict = {}
    running_video_name = None
    inference_time = 0
    for samples, targets in tqdm(data_loader):
        i_iter = i_iter + 1
        video_name = targets[0]['video_name']
        frame_ids = targets[0]['frame_ids']
        center_frame_name = targets[0]['center_frame']
        center_frame_index = frame_ids.index(center_frame_name)
        samples = samples.to(device)
        targets = [{k: v.to(device) if k in ['masks'] else v for k, v in t.items()} for t in targets]
        # ###############################################
        center_gt_path = targets[0]['mask_paths'][center_frame_index]
        center_gt = cv2.imread(center_gt_path, cv2.IMREAD_GRAYSCALE)
        center_gt[center_gt > 0] = 1
        gt_shape = center_gt.shape
        if running_video_name is not None and video_name != running_video_name:
            video_iou = np.mean(list(vid_iou_dict[running_video_name].values()))
            print('video_name:%s iou:%0.3f' % (running_video_name, video_iou))
        running_video_name = video_name
        if aug:
            outputs = compute_predictions_flip_ms(model, samples, targets, gt_shape, ms=True, ms_gather='mean',
                                                  flip=True, flip_gather='mean', clip_tag=video_name,
                                                  scales=[0.7, 0.8, 0.9, 1, 1.1, 1.2])
        else:
            start = time.time()
            outputs = model(samples, clip_tag=video_name)
            inference_time += time.time() - start
            outputs['pred_masks'] = utils.interpolate(outputs['pred_masks'], size=center_gt.shape, mode="bilinear",
                                                      align_corners=False)
            outputs['pred_masks'] = outputs['pred_masks'].sigmoid()
            samples_flipped, targets_flipped = augment_flip(samples, targets)
            outputs_flipped = model(samples_flipped, clip_tag='%s@flipped' % video_name)
            outputs_flipped['pred_masks'] = utils.interpolate(outputs_flipped['pred_masks'], size=gt_shape,
                                                              mode="bilinear", align_corners=False)
            outputs_flipped['pred_masks'] = outputs_flipped['pred_masks'].sigmoid()
            outputs['pred_masks'] = (outputs_flipped['pred_masks'].flip(-1) + outputs['pred_masks']) / 2.0

        src_masks = outputs["pred_masks"]
        yc_logits = src_masks.squeeze(0).cpu().detach().numpy()[center_frame_index, :, :].copy()
        # yc_logits_i = computeWeightedProbs(yc_logits.copy())
        yc_binmask = yc_logits.copy()
        yc_binmask[yc_binmask > 0.5] = 1
        yc_binmask[yc_binmask <= 0.5] = 0
        out = yc_binmask.astype(center_gt.dtype)
        # ########################################
        iou = db_eval_iou(center_gt.copy(), out.copy())
        iou_list.append(iou)
        if video_name not in vid_iou_dict:
            vid_iou_dict[video_name] = {}
        vid_iou_dict[video_name][center_frame_name] = iou
        if save_pred:
            logits_out_dir = os.path.join(out_dir, 'logits', video_name)
            if not os.path.exists(logits_out_dir):
                os.makedirs(logits_out_dir)
            cv2.imwrite(os.path.join(logits_out_dir, '%s.png' % center_frame_name),
                        (yc_logits.astype(np.float32) * 255).astype(np.uint8))
            bm_out_dir = os.path.join(out_dir, 'bin_mask', video_name)
            if not os.path.exists(bm_out_dir):
                os.makedirs(bm_out_dir)
            cv2.imwrite(os.path.join(bm_out_dir, '%s.png' % center_frame_name),
                        (out.astype(np.float32) * 255).astype(np.uint8))  # it is 0, 1
    video_iou = np.mean(list(vid_iou_dict[running_video_name].values()))
    print('video_name:%s iou:%0.3f' % (running_video_name, video_iou))
    print('Inference time ', inference_time / i_iter)
    video_mean_iou = np.mean([np.mean(list(vid_iou_f.values())) for _, vid_iou_f in vid_iou_dict.items()])
    # ### ### Write the results to CSV ### ###
    csv_file_name = 'davis_results.csv'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    video_names = []
    video_ious = []
    for k, v in vid_iou_dict.items():
        vid_iou = np.mean(list(v.values()))
        video_names.append(k)
        video_ious.append('%0.3f' % vid_iou)
        logger.debug('video_name:%s iou:%0.3f' % (k, vid_iou))
    video_names.append('Video Mean')
    video_ious.append('%0.3f' % video_mean_iou)
    with open(os.path.join(out_dir, csv_file_name), 'w') as f:
        cf = csv.writer(f)
        cf.writerow(video_names)
        cf.writerow(video_ious)
    logger.debug('Davis Videos Mean IOU: %0.3f' % video_mean_iou)
    print('Davis Videos Mean IOU: %0.3f' % video_mean_iou)
    return video_mean_iou


def run_inference(args, device, model):
    out_dir = args.output_dir
    if out_dir is None or len(out_dir) == 0:
        out_dir = './results'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # ### Data Loader #########
    if args.dataset == 'davis':
        dataset_val = Davis16ValDataset(num_frames=args.num_frames, val_size=args.val_size,
                                        temporal_strides=args.temporal_strides)
    elif args.dataset == 'moca':
        dataset_val = MoCADataset(num_frames=args.num_frames, min_size=args.val_size,
                                  temporal_strides=args.temporal_strides)
    elif args.dataset == 'ytbo':
        dataset_val = YouTubeObjects(num_frames=args.num_frames, min_size=args.val_size,
                                     temporal_strides=args.temporal_strides)
    else:
        raise ValueError('Dataset not implemented')
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_val = torch.utils.data.BatchSampler(sampler_val, args.batch_size, drop_last=False)
    data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val, collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers)
    with torch.no_grad():
        state_dict = torch.load(args.model_path)['model']
        model.load_state_dict(state_dict)
        if args.dataset == 'moca':
            moca_infer(model, data_loader_val, device, aug=args.aug, save_pred=True, out_dir=out_dir, lprop_mode=args.lprop_mode,
                       compute_memory=args.compute_memory)
            moca_eval(out_dir=out_dir, resize=1)
        elif args.dataset == 'ytbo':
            infer_ytbobj_perseqpercls(model, data_loader_val, device, aug=args.aug, save_pred=args.save_pred,
                                      out_dir=out_dir)
        elif args.dataset == 'davis':
            infer_on_davis(model, data_loader_val, device, save_pred=args.save_pred, aug=args.aug, out_dir=out_dir)
        else:
            raise ValueError('dataset name: %s not implemented' % args.dataset)

def inference_on_all_vos_dataset(args, device, model):
    vos_datasets = ['davis', 'ytbo', 'moca']
    val_sizes = {'davis': 440, 'ytbo': 380, 'moca': 440}
    base_output_dir = args.output_dir
    for data_set_name in vos_datasets:
        args.dataset = data_set_name
        args.val_size = val_sizes[data_set_name]
        args.output_dir = os.path.join(base_output_dir, '%s_%03d_%s' % (data_set_name, args.val_size, 'msc' if args.aug else ''))
        logger.debug('##########################################################')
        logger.debug(args)
        logger.debug('Doing inference on best checkpoint')
        logger.debug(f'Inference on {args.dataset} using val_size:{args.val_size} aug:{args.aug}')
        print('##########################################################')
        print(f'Inference on {args.dataset} using val_size:{args.val_size} aug:{args.aug}')
        run_inference(args, device, model)


def main(args):
    # print(args)
    device = torch.device(args.device)
    utils.init_distributed_mode(args)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """
    The rest part of this function is moved to a new function inference(args, device) for the convenience
    of calling inference of best_checkpoint immediately after training from the train.py.
    See relevant function inference_all_vos_dataset(...) in train.py
    This one saves manual calling of inference.py 3 times for 3 dataset after training.
    """
    model, _ = build_model(args)
    model.to(device)
    if args.dataset == 'all':
        inference_on_all_vos_dataset(args, device, model)
    else:
        run_inference(args, device, model)
    print('Thank You!')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser('VisVOS inference script', parents=[get_args_parser()])
    parsed_args = args_parser.parse_args()
    if not os.path.exists(parsed_args.output_dir):
        os.makedirs(parsed_args.output_dir)

    experiment_name = str(parsed_args.model_path).split('/')[-2]
    print('experiment_name:%s' % experiment_name)

    print('log file: ' + str(os.path.join(parsed_args.output_dir, 'out.log')))  # added by @RK
    logging.basicConfig(
        filename=os.path.join(parsed_args.output_dir, 'out.log'),
        format='%(asctime)s %(levelname)s %(module)s-%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger.debug(parsed_args)
    print(parsed_args)
    main(parsed_args)
