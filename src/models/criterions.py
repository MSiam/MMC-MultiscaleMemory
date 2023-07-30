"""
VisTR criterion classes.
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.util.misc import (nested_tensor_from_tensor_list, interpolate)


def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # import ipdb; ipdb.set_trace()
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.sum()
    # return 5*torch.log(torch.cosh(loss)).sum()


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    # import ipdb; ipdb.set_trace()
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()


class SetCriterion(nn.Module):
    """ This class computes the loss for our model.
    The code is based on the code from VisTR.
    """

    def __init__(self, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = 1
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.L1Loss = torch.nn.L1Loss()

    def loss_masks(self, outputs, targets):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        # import ipdb; ipdb.set_trace()
        src_masks = outputs["pred_masks"]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(target_masks, split=False).decompose()
        target_masks = target_masks.to(src_masks)
        # print('loss_masks->src_masks_shape:%s target_masks_shape:%s'%(str(src_masks.shape), str(target_masks.shape)))
        # import ipdb; ipdb.set_trace()
        num_frames = src_masks.shape[1]
        num_instances = target_masks.shape[1] // num_frames
        if num_instances > 1:
            gt = target_masks[:, 0:num_frames, :, :]
            for i in range(1, num_instances):
                ins_i = target_masks[:, i * num_frames:(i + 1) * num_frames, :, :]
                gt = gt + ins_i
            gt[gt > 0.5] = 1
            target_masks = gt
        # upsample predictions to the target size
        target_size = target_masks.shape[-2:]
        src_masks = interpolate(src_masks, size=target_size, mode="bilinear", align_corners=False)
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)
        # import ipdb;
        # ipdb.set_trace()
        focal_loss_ = sigmoid_focal_loss(src_masks, target_masks)
        dice_loss_ = dice_loss(src_masks, target_masks)
        # l1_loss = self.L1Loss(src_masks.sigmoid(), target_masks)

        if "aux_pred_masks" in outputs:
            # import ipdb; ipdb.set_trace()
            aux_predictions = outputs["aux_pred_masks"]
            if not type(aux_predictions) is list:
                aux_predictions = [aux_predictions]
            w_main = 0.8  # 80% main loss
            w_aux = 0.2 # 20% aux loss
            w_aux_stages = list(range(1, len(aux_predictions) + 1))
            w_aux_sum = sum(w_aux_stages)
            w_aux_stages = [wa * w_aux / w_aux_sum for wa in w_aux_stages]
            focal_loss_ = focal_loss_ * w_main
            dice_loss_ = dice_loss_ * w_main
            for i, aux_pred in enumerate(aux_predictions):
                aux_pred = interpolate(aux_pred, size=target_size, mode="bilinear", align_corners=False)
                aux_pred = aux_pred.flatten(1)
                aux_focal_loss_i = sigmoid_focal_loss(aux_pred, target_masks)
                aux_dice_loss_i = dice_loss(aux_pred, target_masks)
                focal_loss_ = focal_loss_ + aux_focal_loss_i * w_aux_stages[i]
                dice_loss_ = dice_loss_ + aux_dice_loss_i * w_aux_stages[i]
        losses = {
            "loss_mask": focal_loss_,  # + 0.2 * l1_loss,
            "loss_dice": dice_loss_,
        }
        return losses

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = {}
        losses.update(self.loss_masks(outputs, targets))
        return losses
