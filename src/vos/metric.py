import numpy as np


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


# Collected from Isma
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
