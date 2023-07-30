#!/usr/bin/env python

"""
Based on:
# https://github.com/MSiam/motion_adaptation/blob/master/crf/crf_davis.py
"""
import sys
# import cv2
# from imageio import imread as imread
# from imageio import imsave as imsave
from PIL import Image
import glob
import pydensecrf.densecrf as dcrf
import numpy as np
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_softmax
import os
from joblib import Parallel, delayed

import inference as inference
from src.datasets import path_config as dataset_path_config

imgs_path = dataset_path_config.davis16_rgb_path
annots_path = dataset_path_config.davis16_gt_path


def mkdir_p(d):
    try:
        os.makedirs(d)
    except OSError as err:
        if err.errno != 17:
            raise


def apply_crf(img, anno_rgb):
    anno_rgb = np.power(anno_rgb, 1.3)
    min_val = np.min(anno_rgb.ravel())
    max_val = np.max(anno_rgb.ravel())
    out = (anno_rgb.astype('float') - min_val) / (max_val - min_val)

    labels = np.zeros((2, img.shape[0], img.shape[1]))
    labels[1, :, :] = out
    labels[0, :, :] = 1 - out

    img = np.ascontiguousarray(img)
    pred = np.ascontiguousarray(labels)  # pred.swapaxes(0, 2).swapaxes(1, 2))

    d = dcrf.DenseCRF2D(854, 480, 2)  # width, height, nlabels
    unaries = unary_from_softmax(pred, scale=1.0)
    d.setUnaryEnergy(unaries)

    d.addPairwiseGaussian(sxy=0.220880737269, compat=1.24845093352)
    d.addPairwiseBilateral(sxy=22.3761305044, srgb=7.70254062277, rgbim=img, compat=1.40326787165)
    processed = d.inference(15)
    res = np.argmax(processed, axis=0).reshape(480, 854)

    return res


def do_seq(seq, preds_path, save=True):
    # print('current seq:%s' % seq)
    # import ipdb; ipdb.set_trace()
    files = sorted(glob.glob(preds_path + "/%s/*.png"%seq))

    iou_list_after = []
    for f in files:
        im_path = os.path.join(imgs_path, seq, f.split('/')[-1].replace('.png', '.jpg'))
        gt_path = os.path.join(annots_path, seq, f.split('/')[-1])
        out_dir = os.path.join("/".join(f.split("/")[:-3]), f.split("/")[-3] + '_crf_835', f.split("/")[-2])

        # import ipdb; ipdb.set_trace()
        im = imread(im_path)
        gt = imread(gt_path)
        gt = (gt.copy().astype(np.float32) / 255.0).astype(np.uint8)  # 0, 1
        pred = np.array(Image.open(f))
        w_pred = pred.copy()
        w_pred = w_pred.astype(np.float32) / 255.0

        res = apply_crf(im, w_pred)
        iou_crf = inference.db_eval_iou(gt, res)
        iou_list_after.append(iou_crf)
        if save:
            mkdir_p(out_dir)
            out_path = os.path.join(out_dir, f.split('/')[-1])
            imsave(out_path, (res*255).astype(np.uint8))
    return np.mean(iou_list_after)


def seq_eval_only(seq, preds_path, save=False):
    # import ipdb; ipdb.set_trace()
    files = sorted(glob.glob(preds_path + "/%s/*.png"%seq))
    iou_list = []
    for f in files:
        gt_path = os.path.join(annots_path, seq, f.split('/')[-1])
        out_dir = os.path.join("/".join(f.split("/")[:-3]), f.split("/")[-3] + '_bin_mask', f.split("/")[-2])
        pred = np.array(Image.open(f))
        gt = imread(gt_path)
        gt = (gt.copy().astype(np.float32) / 255.0).astype(np.uint8)  # 0, 1
        w_pred = pred.copy().astype(np.float32) / 255.0
        # w_pred = inference.computeWeightedProbs(w_pred.copy())  # use this or not, not a big difference
        bm = np.zeros(w_pred.shape, dtype=np.uint8)
        bm[w_pred > 0.5] = 1
        iou = inference.db_eval_iou(gt, bm)
        iou_list.append(iou)
        if save:
            mkdir_p(out_dir)
            out_path = os.path.join(out_dir, f.split('/')[-1])
            imsave(out_path, (bm*255).astype(np.uint8))
    return np.mean(iou_list)


def main():
    """
    python crf_davis16.py path/to/predition save_all
    arg 1: prediction logits path
    arg 2: save type [save_all, save_crf]
    """
    print('Running ...')
    assert len(sys.argv) >= 2
    pred_path_prefix = sys.argv[1]
    print('logits path: %s'%pred_path_prefix)
    if not os.path.exists(pred_path_prefix):
        raise ValueError('Logits path is not available...')
    
    _save_nocrf_mask = False
    _save_crf_mask = False 
    if len(sys.argv)>2:
        save_type = sys.argv[2]
        print('save_type in argv[2]:%s'%save_type)
        if save_type == 'save_all':
            _save_no_crf_mask = True
            _save_crf_mask = True
        if save_type == 'save_crf':
            _save_crf_mask = True
    print('_save_nocrf_mask:%s _save_crf_mask:%s'%(str(_save_nocrf_mask), str(_save_crf_mask)))

    DAVIS_seqs = ["blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout", "car-shadow", "cows",
                   "dance-twirl", "dog", "drift-chicane", "drift-straight", "goat", "horsejump-high", "kite-surf",
                  "libby", "motocross-jump", "paragliding-launch", "parkour", "scooter-black", "soapbox"]
    seqs = DAVIS_seqs

    # seq_eval_only(seqs[0], pred_path_prefix, save=True)
    # do_seq(seqs[0], pred_path_prefix, save=True)

    seq_iouList_b4crf = Parallel(n_jobs=20)(delayed(seq_eval_only)(seq, pred_path_prefix, save=_save_nocrf_mask) for seq in seqs)
    print('Overall video iou mean from file:%0.3f' % (np.mean(seq_iouList_b4crf)))

    seq_iouList_crf = Parallel(n_jobs=20)(delayed(do_seq)(seq, pred_path_prefix, save=_save_crf_mask) for seq in seqs)
    print('Overall video iou mean after_crf:%0.3f' % (np.mean(seq_iouList_crf)))


if __name__ == "__main__":
    main()
