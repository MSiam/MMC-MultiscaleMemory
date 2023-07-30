
###################################################
# Functions to obtain statistical information
# Coded by Isma Hadji (hadjisma@cse.yorku.ca)
# and Soo Min Kang (kangsoo@cse.yorku.ca)
# Date: July 31, 2019
###################################################
""" Implementation of various statistics functions for quantitative evaluation
e.g: obtain IoU between predicted mask and the groundtruth
"""

import numpy as np
from sklearn.metrics import auc
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import kurtosis, skew


cfg_EPSILON = np.spacing(1, dtype=np.float32)


def getAUROC3D(SalMap3D, GTMask3D, thresholds=np.linspace(0, 1, 256)):
    # compute the area under the ROC curve

    [height, width, duration] = np.shape(GTMask3D)
    counter = 0;
    TPR = [];
    FPR = []
    for tau in thresholds:
        TP = np.zeros((height, width), dtype=np.float32)
        FP = np.zeros((height, width), dtype=np.float32)
        TN = np.zeros((height, width), dtype=np.float32)
        FN = np.zeros((height, width), dtype=np.float32)
        for frame in range(duration):
            BinMask2D = (SalMap3D[:, :, frame] >= tau)  # boolean
            GTMask2D = (GTMask3D[:, :, frame] > 0)  # boolean

            TP += np.logical_and(BinMask2D, GTMask2D) * 1  # float32
            FP += np.logical_and(BinMask2D, np.logical_not(GTMask2D)) * 1  # float32
            TN += np.logical_and(np.logical_not(BinMask2D), np.logical_not(GTMask2D)) * 1  # float32
            FN += np.logical_and(np.logical_not(BinMask2D), GTMask2D) * 1  # float32

        TPR_num = np.count_nonzero(TP)
        TPR_den = np.count_nonzero(TP) + np.count_nonzero(FN) + cfg_EPSILON
        TPR.append(TPR_num / TPR_den)

        FPR_num = np.count_nonzero(FP)
        FPR_den = np.count_nonzero(FP) + np.count_nonzero(TN) + cfg_EPSILON
        FPR.append(FPR_num / FPR_den)

        counter += 1

    return auc(FPR, TPR)


def get_Precision2D(Mask, GT):
    # compute precision (i.e. TP/(TP+FP))

    TP = (GT * Mask).sum()
    FP = ((1 - GT) * Mask).sum()
    pre = float(TP) / (float(TP + FP) + cfg_EPSILON)

    return pre


def get_Recall2D(Mask, GT):
    # compute recall (i.e. TP/(TP+FN))

    TP = (GT * Mask).sum()
    FN = ((1 - Mask) * GT).sum()
    rec = float(TP) / (float(TP + FN) + cfg_EPSILON)

    return rec


def get_FMeasure3D(Mask, GT):
    # compute F-Measure (i.e. 2*((pre*rec)/(pre+rec)))
    annot = 0
    F = 0
    duration = GT.shape[2]
    for frame in range(duration):
        if np.any(GT[:, :, frame] != 0):
            pre = get_Precision2D(Mask[:, :, frame], GT[:, :, frame])
            rec = get_Recall2D(Mask[:, :, frame], GT[:, :, frame])
            F = F + (2 * ((pre * rec) / (pre + rec)))
            annot += 1
    F = F / float(annot)
    return F


def get_IOU2D(Mask1, Mask2):
    # compute the IoU score between two 2D masks
    # where MASK2 is the ground truth mask

    # intersection of Mask1 and Mask2
    I = np.logical_and(Mask1 > 0, Mask2 > 0)

    # union of Mask1 and Mask2
    U = np.logical_or(Mask1 > 0, Mask2 > 0)

    # calculate the numerator and the denominators
    num = np.count_nonzero(I)
    den = np.count_nonzero(U)
    cfg_EPSILON = np.spacing(1, dtype=np.float32)
    if den == 0:  # den <= cfg_EPSILON:
        iou = 1
    else:
        iou = float(num) / float(den)

    return iou


def get_IOU3D(Mask1_3D, Mask2_3D):
    # compute the IoU score between two 3D masks
    # ASSUMING Mask2_3D is the provided GT mask
    duration = Mask1_3D.shape[2]

    ious = np.zeros((duration,), dtype=np.float32)
    annot = 0
    for frame in range(duration):
        ious[frame] = get_IOU2D(Mask1_3D[:, :, frame], Mask2_3D[:, :, frame])
        if np.any(Mask2_3D[:, :, frame] != 0):
            annot += 1
    if ious.sum()>0 and annot>0:
        iou3D = ious.sum() / float(annot)
    else:
        iou3D = np.mean(ious)
    return iou3D


def get_IOU3D_detail(mask3D, annotation3D):
    # compute the IoU score between two 3D masks
    # ASSUMING annotation3D is the provided GT mask
    duration = mask3D.shape[2]

    iou_detail = np.zeros((duration,), dtype=np.float32)
    annot = 0
    # import ipdb;ipdb.set_trace()
    for frame in range(duration):
        iou_detail[frame] = get_IOU2D(mask3D[:, :, frame], annotation3D[:, :, frame])
        if np.any(annotation3D[:, :, frame] != 0):
            annot += 1
    # import ipdb;ipdb.set_trace()
    if iou_detail.sum() > 0 and annot > 0:
        iou3D = iou_detail.sum() / float(annot)
    else:
        iou3D = np.mean(iou_detail)
    return iou3D, iou_detail


def get_temporal_weight(duration, frame, lambdaA=0.01):
    """
    Copied from Ismas' code for vos: Sal2Bin.get_weight
    """
    ts = np.arange(duration)[:, np.newaxis]  # [duration,1]
    weight = (pairwise_distances(ts, metric='euclidean')) ** 2
    weight = np.exp(-lambdaA * weight)  # [duration, duration]
    weight_vec = weight[:, frame]  # [duration,1]
    return weight_vec


def getCoMass2D(SalMap2D, OutputType='unnormalized'):
    # compute the centre of mass given a 2D saliency map
    # input: SalMap2D is [height,width] array containing real values and
    #       OutputType specifies the type of output; can be {'unnormalized',normalized'}
    # output: [xc, yc] are real-valued coordinates.

    [height, width] = np.shape(SalMap2D)

    # define the (x,y)-coordinate system
    if OutputType == 'unnormalized':
        xs = np.linspace(1, width, width, dtype=np.float32)
        ys = np.linspace(1, height, height, dtype=np.float32)
    elif OutputType == 'normalized':
        xs = np.linspace(0, 1, width, dtype=np.float32)
        ys = np.linspace(0, 1, height, dtype=np.float32)
    [X, Y] = np.meshgrid(xs, ys)

    # replace NaNs
    SalMap2D = np.nan_to_num(SalMap2D)

    # calculate the centre of mass in the x-direction
    xc = np.sum(np.sum(SalMap2D * X, axis=1), axis=0)
    xc = xc / np.sum(np.sum(SalMap2D, axis=1), axis=0)

    # calculate the centre of mass in the y-direction
    yc = np.sum(np.sum(SalMap2D * Y, axis=1), axis=0)
    yc = yc / np.sum(np.sum(SalMap2D, axis=1), axis=0)

    return [xc, yc]


def getVar2D(SalMap2D, xc, yc, OutputType):
    # calculate the variance given a 2D saliency map
    # input: SalMap2D is a [height,width] array containing real values
    #       [xc, yc] are the centre of mass coordinates from getCoMass2D
    #       OutputType specifies the type of output; can be {'unnormalized',normalized'}
    # output: [varX,varY] is the variance in the x- and y-direction, and
    # varMag is the magnitude of (varX,varY) (i.e. varMag = sqrt(varX^2 + varY^2)).

    [height, width] = np.shape(SalMap2D)

    # define the (x,y)-coordinate system
    if OutputType == 'unnormalized':
        xs = np.linspace(1, width, width, dtype=np.float32)
        ys = np.linspace(1, height, height, dtype=np.float32)
    elif OutputType == 'normalized':
        xs = np.linspace(0, 1, width, dtype=np.float32)
        ys = np.linspace(0, 1, height, dtype=np.float32)
    [X, Y] = np.meshgrid(xs, ys)

    # count the number of nonzero and non-NaN values in SalMap
    tmp = np.logical_and(~np.isnan(SalMap2D), SalMap2D != 0)
    NumVals = np.count_nonzero(tmp)

    # replace NaNs with 0s
    SalMap2D = np.nan_to_num(SalMap2D)

    # calculate the variance in the x-direction
    varX = np.sum(np.sum(SalMap2D * (X - xc) ** 2, axis=1, dtype=np.float32), axis=0, dtype=np.float32)
    varX = varX / (float(NumVals) + np.spacing(1))

    # calculate the variance in the y-direction
    varY = np.sum(np.sum(SalMap2D * (Y - yc) ** 2, axis=1, dtype=np.float32), axis=0, dtype=np.float32)
    varY = varY / (float(NumVals) + np.spacing(1))

    # calculate the magnitude of the variance
    varMag = np.sqrt(varX ** 2 + varY ** 2)

    return [varX, varY, varMag]


def getCenterednessCompactness(SalMap3D):
    """
    Copied from Ismas' code for vos: Util.getCenterednessCompactness 
    """
    # function to estimate a combination of compactness and centeredness weighted by temporal centeredness
    height, width, duration = SalMap3D.shape
    middleFrame = int(np.ceil(float(duration) / 2.0))

    # STEP 1: get weights estimating distance from center of video
    lambdaT = (18. / (duration ** 2))  # normalizing factor for the "rate of decay" of temporal weights
    temporal_weight = get_temporal_weight(duration, middleFrame, lambdaA=lambdaT)

    # STEP 2: Get the max distance to the center for normalization purposes
    mH = int(np.ceil(float(height) / 2.0))
    mW = int(np.ceil(float(width) / 2.0))
    MaxdistC = np.sqrt((mW) ** 2 + (mH) ** 2)

    # STEP 3: calculate the max possible variance for normalization purposes
    MaxSalMap2D = np.ones((height, width), dtype=np.float32)
    Mxc, Myc = getCoMass2D(MaxSalMap2D, 'unnormalized')
    _, _, VarMagMax = getVar2D(MaxSalMap2D, Mxc, Myc, 'unnormalized')

    # STEP 4:  calculate the centre of mass distance and variance per frame
    cntr = np.zeros((duration,), dtype=np.float32)
    cmpt = np.zeros((duration,), dtype=np.float32)
    cntrcmpt = np.zeros((duration,), dtype=np.float32)
    # import ipdb;ipdb.set_trace()
    for frame in range(duration):
        SalMap2D = SalMap3D[:, :, frame]
        # calculate the centre of mass
        xc, yc = getCoMass2D(SalMap2D, 'unnormalized')
        # calculate distance to center
        UndistC = np.sqrt((xc - mW) ** 2 + (yc - mH) ** 2)
        # normalize the centeredness score
        cntr[frame] = UndistC / MaxdistC

        # calculate the variance
        _, _, VarMag = getVar2D(SalMap2D, xc, yc, 'unnormalized')
        # normalize the compactness score
        cmpt[frame] = VarMag / VarMagMax

        # STEP 5: Combine the centeredness and compactness measures using the temporal weights
        cntrcmpt[frame] = (temporal_weight[frame] * cntr[frame]) + ((1 - temporal_weight[frame]) * cmpt[frame])

    return cntr, cmpt, cntrcmpt


def get_normalized_intensity_difference(feat_map, gt_mask):
    """
    :param feat_map: feature map of shape TxHxW
    :param gt_mask: gt-mask of shape TxHxW
    :return:
    """
    # import ipdb;ipdb.set_trace()
    ft = feat_map.copy()
    gt = gt_mask.copy()
    if len(gt.shape) == 4:
        gt = gt[:, :, :, 0]
    if len(ft.shape) == 4:
        ft = np.max(ft, axis=3, keepdims=False)
    if len(ft.shape) != len(gt.shape):
        raise Exception('feat and gt shape mismatch')

    T, gth, gtw = gt.shape
    if gt.shape[1] != ft.shape[1] or gt.shape[2] != ft.shape[2]:
        ft = ft[:, :, :, np.newaxis]
        ft = tf.image.resize(ft, (gth, gtw))
        ft = np.add(ft[:, :, :, 0], 0)
    gt_max = np.max(gt)
    if gt_max != 0:
        gt = (gt * 1.0) / gt_max
        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0

    pixel_per_frame = gth * gtw
    n_cells = np.ones((T,)) * pixel_per_frame
    fg_mask = gt > 0.5
    fg_sz = np.sum(fg_mask, axis=(1, 2))
    bk_sz = n_cells - fg_sz
    fg_ftsum = np.sum(ft * gt, axis=(1, 2))
    bk_ftsum = np.sum(ft * (1 - gt), axis=(1, 2))
    fg_ftsum = fg_ftsum / fg_sz
    bk_ftsum = bk_ftsum / bk_sz
    fg_ftsum = np.nan_to_num(fg_ftsum)
    bk_ftsum = np.nan_to_num(bk_ftsum)
    ft_diff = np.abs(fg_ftsum - bk_ftsum)
    ft_diff = np.round(ft_diff * 100) / 100  # keep 2 decimal point
    return ft_diff


# NEEDED
def BimodalityCoeff(SalMap2D):
    # function to estimate bimodality of a distribution
    # > 0.6 is likely bimodal (or NOT unimodal)
    # For details See: https://en.wikipedia.org/wiki/Multimodal_distribution#Mixture_of_two_normal_distributions
    if SalMap2D.ndim == 2:
        height, width = SalMap2D.shape
        data = np.reshape(SalMap2D, (height * width, 1))
    elif SalMap2D.ndim == 1:
        data = SalMap2D
    n = data.shape[0]

    if n <= 3:
        b = 0.0
    else:
        nn = 3. * ((n - 1) ** 2) / ((n - 2) * (n - 3))
        s = skew(data)
        k = kurtosis(data)

        b = (s ** 2 + 1) / (k + nn)
    return b


# NEEDED
def CompAreasL1(area1, area2):
    areaDiff = np.abs(area1 - area2).mean()
    return areaDiff
