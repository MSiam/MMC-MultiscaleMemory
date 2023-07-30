"""
Training script for ResNet+Transformer for VOS
Based on training script of VisTR (https://github.com/Epiphqny/VisTR)
Which was modified from DETR (https://github.com/facebookresearch/detr)
"""
import argparse
import csv
import datetime
import time
from pathlib import Path
import math
import os
import sys
from typing import Iterable
import numpy as np
import cv2
import random
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, DistributedSampler
# from src.util.torch_poly_lr_decay import PolynomialLRDecayWithOsc as PolynomialLRDecay
from src.util.torch_poly_lr_decay import PolynomialLRDecay as PolynomialLRDecay
from src.datasets.davis.davis16_ytvos_data_loader_v4 import Davis16_YTVOS_DataLoaderV2 as Davis16_DATASET
# from src.datasets.test.youtube_objects import YouTubeObjects
import src.util.misc as utils
import src.datasets.transforms as T
import wandb
from src.util.wandb_utils import init_or_resume_wandb_run, get_viz_img
import src.vos.metric as vos_metric
import pathlib
import torch.backends.cudnn as cudnn
from src.models.vos.med_vt_swin import build_model_swin_medvt as build_model
from src.models.utils import parse_argdict

from tqdm import tqdm

def get_args_parser():
    args_parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    args_parser.add_argument('--lr', default=1e-4, type=float)
    args_parser.add_argument('--lr_backbone', default=1e-6, type=float)
    args_parser.add_argument('--end_lr', default=1e-6, type=float)
    args_parser.add_argument('--lr_drop', default=4, type=int)
    args_parser.add_argument('--poly_power', default=0.9, type=float)
    args_parser.add_argument('--weight_decay', default=1e-4, type=float)
    args_parser.add_argument('--epochs', default=15, type=int)
    args_parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    args_parser.add_argument('--aux_loss', default=False, help="Use auxiliary decoding losses")
    args_parser.add_argument('--batch_size', default=1, type=int)
    args_parser.add_argument('--train_size', default=300, type=int)
    args_parser.add_argument('--val_size', default=360, type=int)
    args_parser.add_argument('--temporal_strides', default=[1], nargs='+', type=int,
                        help="temporal strides used to construct input cip")
    args_parser.add_argument('--eval_combo', default=False, action='store_true')
    args_parser.add_argument('--lprop_mode', default=0, type=int, help='Choose 1 for unidir lprop, 2 for bidir lprop, 0 for no lprop')
    args_parser.add_argument('--feat_loc', default='late', type=str, help='Choose early or late features to compare in lprop')
    args_parser.add_argument('--pretrain_settings', default=None, nargs=argparse.REMAINDER)
    args_parser.add_argument('--stacked_lprop', type=int, default=1, help="repeat the lprop")

    # args_parser.add_argument('--experiment_name', default='exp_007d_{params_summary}_msms_60c_post_cross_blnr_use_custom_inst_norm')
    args_parser.add_argument('--experiment_name',
                             default='absx_001e2_msms_soft_{params_summary}')

    args_parser.add_argument('--output_dir',
                             default='outputs/medvt2/',
                             help='path where to save')
    args_parser.add_argument('--use_wandb', action='store_true')
    args_parser.add_argument('--wandb_user', type=str, default='yvv')
    args_parser.add_argument('--wandb_project', type=str, default='transformervos')
    args_parser.add_argument('--viz_freq', type=int, default=2000)
    args_parser.add_argument('--viz_train_img_freq', type=int, default=-1)
    args_parser.add_argument('--viz_val_img_freq', type=int, default=-1)

    # Model parameters
    args_parser.add_argument('--resnet50_coco_weights_path', type=str,
                             default="/local/riemann/home/rezaul/projects/medvt2-main/pretrained/detr_coco/384_coco_r50.pth",
                             help="Path to the pretrained model.")
    args_parser.add_argument('--resnet101_coco_weights_path', type=str,
                             default="/local/riemann/home/rezaul/projects/medvt2-main/pretrained/detr_coco/384_coco_r101.pth",
                             help="Path to the pretrained model.")
    args_parser.add_argument('--swin_s_pretrained_path', type=str,
                             default="/local/riemann/home/rezaul/projects/medvt2-main/pretrained/swin_small_patch244_window877_kinetics400_1k.pth",
                             help="swin-s pretrained model path.")
    args_parser.add_argument('--swin_b_pretrained_path', type=str,
                             default="/local/riemann/home/rezaul/projects/medvt2-main/pretrained/swin_base_patch244_window877_kinetics400_22k.pth",
                             help="swin-s pretrained model path.")
    args_parser.add_argument('--swin_b_pretrained_path_k600', type=str,
                             default="/local/riemann/home/rezaul/projects/medvt2-main/pretrained/swin_base_patch244_window877_kinetics600_22k.pth",
                             help="swin-s pretrained model path.")

    # * Backbone
    args_parser.add_argument('--backbone', default='swinB', type=str,
                             help="backbone to use, [resnet50, resnet101, swinS, swinB]")
    args_parser.add_argument('--model_type', default='encoder-decoder', type=str,
                             help="backbone to use, [encoder-decoder, vanilla-fpn, vanilla]")
    args_parser.add_argument('--use_fpn', default='True', type=bool,
                             help="to use fpn or not")

    args_parser.add_argument('--dilation', default=[False, False, False], action='store_true',
                             help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    args_parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                             help="Type of positional embedding to use on top of the image features")

    # * Transformer
    args_parser.add_argument('--enc_layers', nargs='+', default=(6,), type=int,
                             help="Number of encoding layers in the transformer")
    args_parser.add_argument('--enc_separable_attn', default='thw', type=str,
                             help="Encoder separable attn , (thw, t-hw, th-tw, th-tw-hw )")
    args_parser.add_argument('--encoder_cross_layer', default=False, type=bool,
                             help="Cross resolution attention")
    args_parser.add_argument('--use_bilinear', action="store_true",
                             help="flag to use bilinear in cross res attention")

    args_parser.add_argument('--dec_layers', default=9, type=int,
                             help="Number of decoding layers in the transformer")
    args_parser.add_argument('--dim_feedforward', default=1024, type=int,
                             help="Intermediate size of the feedforward layers in the transformer blocks")
    args_parser.add_argument('--hidden_dim', default=384, type=int,
                             help="Size of the embeddings (dimension of the transformer)")
    args_parser.add_argument('--dropout', default=0.1, type=float,
                             help="Dropout applied in the transformer")
    args_parser.add_argument('--nheads', default=8, type=int,
                             help="Number of attention heads inside the transformer's attentions")

    args_parser.add_argument('--decoder_type', type=str, default='multiscale_query')
    args_parser.add_argument('--num_frames', default=8, type=int,
                             help="Number of frames")
    args_parser.add_argument('--num_queries', default=8, type=int,
                             help="Number of query sots")
    args_parser.add_argument('--pre_norm', action='store_true')

    args_parser.add_argument('--use_soft_mask_encoder', default=False, type=bool,
                             help="soft mask in the encoder")

    args_parser.add_argument('--use_mem_mask', default=False, type=bool,
                             help="prev mask in inference")

    args_parser.add_argument('--mask_consistency_decoder', default=False, type=bool,
                             help="discarded, too much memory, instead we will use label propagation")

    # * Segmentation
    args_parser.add_argument('--masks', default=True, action='store_true',
                             help="Train segmentation head if the flag is provided")

    # * Matcher
    args_parser.add_argument('--set_cost_class', default=1, type=float,
                             help="Class coefficient in the matching cost")
    args_parser.add_argument('--set_cost_bbox', default=5, type=float,
                             help="L1 box coefficient in the matching cost")
    args_parser.add_argument('--set_cost_giou', default=2, type=float,
                             help="giou box coefficient in the matching cost")
    # * Loss coefficients
    args_parser.add_argument('--mask_loss_coef', default=1, type=float)
    args_parser.add_argument('--dice_loss_coef', default=1, type=float)
    args_parser.add_argument('--bbox_loss_coef', default=5, type=float)
    args_parser.add_argument('--giou_loss_coef', default=2, type=float)
    args_parser.add_argument('--eos_coef', default=0.1, type=float,
                             help="Relative classification weight of the no-object class")

    args_parser.add_argument('--remove_difficult', action='store_true')
    args_parser.add_argument('--finetune', action='store_true')

    args_parser.add_argument('--device', default='cuda',
                             help='device to use for training / testing')
    args_parser.add_argument('--seed', default=42, type=int)
    args_parser.add_argument('--resume', default=False, help='resume from checkpoint')
    # args_parser.add_argument('--resume', default='./outputs/vos_res101s32_vistr_cls3d/checkpoint0003.pth', help='resume from checkpoint')

    args_parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                             help='start epoch')
    args_parser.add_argument('--eval', action='store_true')
    args_parser.add_argument('--num_workers', default=4, type=int)

    # distributed training parameters
    args_parser.add_argument('--world_size', default=1, type=int,
                             help='number of distributed processes')
    args_parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args_parser.add_argument('--compute_memory', action='store_true')
    return args_parser


def record_csv(filepath, row):
    with open(filepath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    return


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    output_viz_dir=Path('./outputs/'), use_wandb: bool = False,
                    viz_freq: int = 1000, total_epochs=15, args=None):
    # import ipdb; ipdb.set_trace()
    inverse_norm_transform = T.InverseNormalizeTransforms()
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train Epoch: [{}/{}]:'.format(epoch, total_epochs)
    print_freq = 2000
    i_iter = 0
    if not os.path.exists(output_viz_dir):
        os.makedirs(output_viz_dir)
    _loss_t_csv_fn = os.path.join(output_viz_dir, 'loss.csv')
    if epoch == 0 and os.path.exists(_loss_t_csv_fn):
        os.rename(_loss_t_csv_fn, os.path.join(output_viz_dir, 'loss_{}.csv'.format(time.time())))
    loss_sum = 0
    item_count = 0
    tt1 = time.time()
    # import ipdb; ipdb.set_trace()
    for samples, targets in metric_logger.log_every(tqdm(data_loader), print_freq, header):
        i_iter = i_iter + 1
        if i_iter > 0 and i_iter % print_freq == 0:
            tt2 = time.time()
            dt_tm = str(datetime.timedelta(seconds=int(tt2 - tt1)))
            logger.debug('{} i_iter_Training_time:{} '.format(i_iter, dt_tm))
            # import ipdb; ipdb.set_trace()
            if False and hasattr(model, 'transformer') and args.use_soft_mask_encoder:
                logger.debug('model.transformer.encoder.layers[0][0].soft_mask_sigma: {:e} soft_mask_alpha:{:e}'.format(
                    model.transformer.encoder.layers[0][0].soft_mask_sigma.max().item(),
                    model.transformer.encoder.layers[0][0].soft_mask_alpha.max().item()
                ))
                logger.debug(
                    'model.transformer.encoder.cross_res_layers[0].soft_mask_sigma: {:e} soft_mask_alpha:{:e}'.format(
                        model.transformer.encoder.cross_res_layers[0].soft_mask_sigma.item(),
                        model.transformer.encoder.cross_res_layers[0].soft_mask_alpha.item()
                    ))
                logger.debug('model.transformer.encoder.layers[1][0].soft_mask_sigma: {:e} soft_mask_alpha:{:e}'.format(
                    model.transformer.encoder.layers[1][0].soft_mask_sigma.max().item(),
                    model.transformer.encoder.layers[1][0].soft_mask_alpha.max().item()
                ))

        # import ipdb; ipdb.set_trace()
        samples = samples.to(device)
        targets = [{k: v.to(device) if k in ['masks'] else v for k, v in t.items()} for t in targets]
        outputs = model(samples)
        # import  ipdb;ipdb.set_trace()
        loss_dict = criterion(outputs, targets)
        # loss2 = criterion2(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            logger.critical("Loss is {}, skip training for this sample".format(loss_value))
            logger.critical(loss_dict_reduced)
            logger.debug('video_name: {} frame_ids:{} center_frame:{}'.format(targets[0]['video_name'],
                                                                              str(targets[0]['frame_ids']),
                                                                              targets[0]['center_frame']))
            continue
            # sys.exit(1)
        optimizer.zero_grad()
        # import ipdb; ipdb.set_trace()
        # if targets[0]['dataset'] != 'davis':
        #    losses = losses * (math.exp(-(epoch / 10)))
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # import ipdb; ipdb.set_trace()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if use_wandb:
            wandb_dict = {'loss': loss_value, 'lr': optimizer.param_groups[0]["lr"]}
            viz_img = get_viz_img(samples.tensors, targets, outputs, inverse_norm_transform)
            if i_iter % viz_freq == 0:
                wandb_dict['viz_img'] = wandb.Image(viz_img)
            wandb.log(wandb_dict)
        # if utils.is_main_process() and args.viz_train_img_freq > 0 and i_iter % args.viz_train_img_freq == 0:
        #    save_train_viz(output_viz_dir, ' train', epoch, i_iter, samples.tensors, inverse_norm_transform, targets,
        #                   outputs)
        loss_sum += float(loss_value)
        item_count += 1
        if i_iter % 50 == 49:
            loss_avg = loss_sum / item_count
            loss_sum = 0
            item_count = 0
            record_csv(_loss_t_csv_fn, ['%e' % loss_avg])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.debug("Averaged stats:{}".format(metric_logger))
    # save_loss_plot(epoch, _loss_t_csv_fn, viz_save_dir=output_viz_dir)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, epoch: int, output_viz_dir=None, total_epochs=15, use_wandb=False):
    # import ipdb; ipdb.set_trace()
    # inverse_norm_transform = T.InverseNormalizeTransforms()
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test Epoch: [{}/{}]:'.format(epoch, total_epochs)
    i_iter = 0
    running_video_name = None
    iou_dict = {}
    video_info = {}
    for samples, targets in metric_logger.log_every(tqdm(data_loader), 500, header):
        i_iter = i_iter + 1
        video_name = targets[0]['video_name']
        center_frame = targets[0]['center_frame']
        frame_ids = targets[0]['frame_ids']
        vid_len = targets[0]['vid_len']
        center_frame_index = frame_ids.index(center_frame)
        video_info[video_name] = vid_len
        # #######################
        center_gt_path = targets[0]['mask_paths'][center_frame_index]
        center_gt = cv2.imread(center_gt_path, cv2.IMREAD_GRAYSCALE)
        center_gt[center_gt > 0] = 1
        # #######################
        samples = samples.to(device)
        targets = [{k: v.to(device) if k in ['masks'] else v for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            logger.critical("Loss is {}, stopping training".format(loss_value))
            logger.critical(loss_dict_reduced)
            sys.exit(1)
        # import ipdb; ipdb.set_trace()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # if utils.is_main_process() and i_iter % 10 == 9:
        #    save_train_viz(output_viz_dir, 'val', epoch, i_iter, samples.tensors, inverse_norm_transform, targets,
        #                   outputs)
        # import ipdb; ipdb.set_trace()
        # ###################################
        src_masks = outputs["pred_masks"]
        src_masks = utils.interpolate(src_masks, size=center_gt.shape, mode="bilinear", align_corners=False)
        #src_masks = src_masks.sigmoid()
        yc = src_masks.squeeze(0).cpu().detach().numpy().copy()
        yc[yc > 0.5] = 1
        yc[yc <= 0.5] = 0
        out = yc[center_frame_index, :, :].astype(center_gt.dtype)
        # ########################################
        iou = vos_metric.db_eval_iou(center_gt.copy(), out.copy())
        if use_wandb:
            wandb_dict = {'val_loss': loss_value, 'val_iou': iou}
            wandb.log(wandb_dict)
        #########################################
        # import ipdb; ipdb.set_trace()
        if running_video_name is None or running_video_name != video_name:
            # if running_video_name is not None:
            # logger.debug(
            #    'running_video_name:{} iou_dict:{}'.format(running_video_name, str(iou_dict[running_video_name])))
            # logger.debug(
            #    'running_video_name:{} video_len:{}'.format(running_video_name, video_info[running_video_name]))
            running_video_name = video_name
            iou_dict[running_video_name] = {}
            # import ipdb; ipdb.set_trace()
        iou_dict[running_video_name][center_frame] = iou
    mean_iou = np.mean([np.mean(list(vid_iou_f.values())) for _, vid_iou_f in iou_dict.items()])
    logger.debug('Test results summary--------------------------------------')
    logger.debug('Epoch:%03d Test mean iou: %0.3f' % (epoch, mean_iou))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.debug("Averaged stats:{}".format(metric_logger))
    video_ious = [np.mean(list(vid_iou_f.values())) for _, vid_iou_f in iou_dict.items()]
    return mean_iou, video_ious


def create_data_loaders(args):
    use_ytvos_for_train = not args.finetune
    dataset_train = Davis16_DATASET(is_train=True, num_frames=args.num_frames, train_size=args.train_size,
                                    use_ytvos=use_ytvos_for_train, soenet_feats=False, soe_feat_type='soenet',
                                    use_flow=False, random_range=False)
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_train.set_epoch(args.start_epoch)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    dataset_val = Davis16_DATASET(is_train=False, num_frames=args.num_frames, val_size=args.val_size,
                                  use_ytvos=False, soenet_feats=False, soe_feat_type='soenet', use_flow=False,
                                  random_range=False)
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_val = torch.utils.data.BatchSampler(
        sampler_val, args.batch_size, drop_last=False)
    data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val, collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers)
    return data_loader_train, data_loader_val


def train(args, device, model, criterion):
    # import ipdb; ipdb.set_trace()
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug('number of params:{}'.format(n_parameters))
    # import ipdb; ipdb.set_trace()
    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": args.lr
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(param_dicts, lr=args.lr,momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=args.epochs-1, end_learning_rate=args.end_lr,
                                     power=args.poly_power)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=[4, 6, 8, 10])
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, min_lr=args.end_lr, verbose=1)

    # ############################################################################
    output_dir = Path(args.output_dir)
    output_viz_dir = output_dir / 'viz'

    if 'pretrained_model_path' in args.pretrain_settings:
        state_dict = torch.load(args.pretrain_settings['pretrained_model_path'], map_location='cpu')
        model_without_ddp.load_state_dict(state_dict['model'], strict=False)
        logger.debug('=======> Loaded Pretrained Weights')

    if args.resume:
        logger.debug('Resuming from checkpoint')
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            resume_path = os.path.join(args.output_dir, 'checkpoint_last.pth')
            if os.path.exists(resume_path):
                checkpoint = torch.load(resume_path, map_location='cpu')
            else:
                checkpoint = None

        if checkpoint is not None:
            model_without_ddp.load_state_dict(checkpoint['model'])
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
                logger.debug('start_epoch:{}'.format(args.start_epoch))

    # ### DATASETS ###################################
    data_loader_train, data_loader_val = create_data_loaders(args)
    start_time = time.time()
    best_eval_iou = 0
    best_eval_epoch = 0
    logger.debug("Start training")
    # logger.debug('Best eval epoch:{} mean_iou:{} '.format(best_eval_epoch, best_eval_iou))
    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        """
        if epoch <1:
            optimizer.param_groups[1]['lr'] = 0
        if epoch == 1:
            optimizer.param_groups[1]['lr'] = args.lr_backbone
        """

        logger.debug('epoch: %3d  optimizer.param_groups[0][lr]: %e' % (epoch, optimizer.param_groups[0]['lr']))
        logger.debug('epoch: %3d  optimizer.param_groups[1][lr]: %e' % (epoch, optimizer.param_groups[1]['lr']))
        train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, output_viz_dir, use_wandb=args.use_wandb,
            viz_freq=args.viz_freq, total_epochs=args.epochs, args=_args)
        print('Finished training', epoch)
        t2 = time.time()
        print('Epoch ', epoch, ' Done Training')
        torch.cuda.empty_cache()
        mean_iou, davis_iou_list = evaluate(
            model, criterion, data_loader_val, device, epoch, output_viz_dir, total_epochs=args.epochs,
            use_wandb=args.use_wandb
        )
        print('Finished evaluation', epoch)
        t2 = time.time()
        print('Epoch ', epoch, ' Done Evaluation')
        #torch.cuda.empty_cache()
        logger.debug('**************************')
        logger.debug('**************************')
        logger.debug('[Epoch:%2d] val_mean_iou:%0.3f' % (epoch, mean_iou))
        if args.use_wandb:
            wandb.log({'miou val': mean_iou})
        if mean_iou > best_eval_iou:
            best_eval_iou = mean_iou
            best_eval_epoch = epoch
        logger.debug('Davis Best eval epoch:%03d mean_iou: %0.3f' % (best_eval_epoch, best_eval_iou))
        if epoch > -1:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_last.pth']
            if epoch == best_eval_epoch:
                checkpoint_paths.append(output_dir / 'checkpoint_best.pth')
            for checkpoint_path in checkpoint_paths:
                logger.debug('saving ...checkpoint_path:{}'.format(str(checkpoint_path)))
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        t3 = time.time()
        train_time_str = str(datetime.timedelta(seconds=int(t2 - t1)))
        eval_time_str = str(datetime.timedelta(seconds=int(t3 - t2)))
        logger.debug(
            'Epoch:{}/{} Training_time:{} Eval_time:{}'.format(epoch, args.epochs, train_time_str, eval_time_str))
        logger.debug('##########################################################')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.debug('Training time {}'.format(total_time_str))
    return model


def main(args):
    print('starting main ...')
    cudnn.benchmark = False
    cudnn.deterministic = True
    seed = args.seed + utils.get_rank()
    # fix the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # import ipdb; ipdb.set_trace()
    utils.init_distributed_mode(args)
    logger.debug("git:\n  {}\n".format(utils.get_sha()))
    device = torch.device(args.device)
    model, criterion = build_model(args)
    logger.debug(str(model))
    model.to(device)
    # ########### MODEL TRAIN #################################
    train(args, device, model, criterion)
    # ########### ##### Test Best Checkpoint ##################
    from inference import inference_on_all_vos_dataset
    args.model_path = Path(args.output_dir) / 'checkpoint_best.pth'
    args.save_pred = True
    logger.debug('##########################################################')
    logger.debug('Inference Single Scale')
    logger.debug('##########################################################')
    args.aug = False
    inference_on_all_vos_dataset(args, device, model)
    logger.debug('##########################################################')
    logger.debug('Inference Multi Scales')
    logger.debug('##########################################################')
    args.aug = True
    inference_on_all_vos_dataset(args, device, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VisTR training and evaluation script', parents=[get_args_parser()])
    _args = parser.parse_args()

    if _args.pretrain_settings is not None:
        _args.pretrain_settings = parse_argdict(_args.pretrain_settings)
    else:
        _args.pretrain_settings = {}

    params_summary = '%s_%s_enc%s_dec%s_%s_t%dv%df%d_lr%0.1e_%0.1e_ep_%02d' % (
        _args.model_type,
        _args.backbone,
        str(_args.enc_layers).replace(',', '_').replace('(', '').replace(')', '').replace(' ', ''),
        str(_args.dec_layers),
        'fpn' if _args.use_fpn else 'nofpn',
        _args.train_size, _args.val_size, _args.num_frames, _args.lr, _args.lr_backbone, _args.epochs
    )
    print('params_summary:%s' % params_summary)
    _args.experiment_name = _args.experiment_name.replace('{params_summary}', params_summary)
    print('parsed_args.experiment_name: %s' % _args.experiment_name)

    output_path = os.path.join(_args.output_dir, _args.experiment_name)
    print(output_path)
    _args.output_dir = output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('log file: ' + str(os.path.join(output_path, 'out.log')))  # added by @RK
    logging.basicConfig(
        filename=os.path.join(output_path, 'out.log'),
        format='%(asctime)s %(levelname)s %(module)s-%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.debug(output_path)
    logger.debug('experiment_name: {}'.format(_args.experiment_name))
    logger.debug("This is our baseline with resnet101s32 and transformer encoder-decoder.")
    logger.debug('Using flip, poly lr, epochs 15, adamw')
    logger.debug("Used Args are {}".format(str(_args)))

    if _args.use_wandb:
        wandb_id_file_path = pathlib.Path(os.path.join(output_path, _args.experiment_name + '_wandb.txt'))
        config = init_or_resume_wandb_run(wandb_id_file_path,
                                          entity_name=_args.wandb_user,
                                          project_name=_args.wandb_project,
                                          run_name=_args.experiment_name,
                                          config=_args)
        logger.debug("Initialized Wandb")

    print(_args)
    logger.debug(_args)
    logger.debug(output_path)
    if _args.output_dir:
        Path(_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(_args)
    logger.debug('Finished training...')
    print('Finished training...')
