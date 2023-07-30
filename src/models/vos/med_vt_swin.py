"""
MED-VT model.
Some parts of the code is taken from VisTR (https://github.com/Epiphqny/VisTR)
which was again Modified from DETR (https://github.com/facebookresearch/detr)
And modified as needed.
"""
import torchvision.ops
import copy
from typing import Optional, List, OrderedDict
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import math
import logging
import os

from src.models import criterions
from src.models.swin.swin_transformer_3d import SwinTransformer3D
from src.models.label_propagation import LabelPropagator
from src.util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from src.util.misc import NestedTensor, is_main_process

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class LRU(OrderedDict):
    'Limit size, evicting the least recently looked-up key when full'

    def __init__(self, maxsize=128, *args, **kwds):
        self.maxsize = maxsize
        super().__init__(*args, **kwds)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, num_frames=36, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.frames = num_frames
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        # import ipdb; ipdb.set_trace()
        x = tensor_list.tensors
        mask = tensor_list.mask
        n, h, w = mask.shape
        mask = mask.reshape(n // self.frames, self.frames, h, w)
        assert mask is not None
        not_mask = ~mask
        z_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos = torch.cat((pos_z, pos_y, pos_x), dim=4).permute(0, 1, 4, 2, 3)
        # import ipdb; ipdb.set_trace()
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 3
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, num_frames=args.num_frames, normalize=True)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        print('dilation:{}'.format(dilation))
        if not type(dilation) is list:
            dilation = [False, False, dilation]
        print('dilation:{}'.format(dilation))
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=dilation,
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, context_dim):
        super().__init__()
        print('Creating FPN-> dim:%d context_dim:%d' % (dim, context_dim))
        inter_dims = [dim, context_dim, context_dim, context_dim, context_dim, context_dim]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.conv_offset = torch.nn.Conv2d(inter_dims[3], 18, 1)  # , bias=False)
        self.dcn = torchvision.ops.DeformConv2d(inter_dims[3], inter_dims[4], 3, padding=1, bias=False)
        self.dim = dim

        for name, m in self.named_modules():
            if name == "conv_offset":
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            else:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, fpns: List[Tensor]):
        multi_scale_features = []
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)
        multi_scale_features.append(x)

        cur_fpn = fpns[0]
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)
        multi_scale_features.append(x)

        cur_fpn = fpns[1]
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)
        multi_scale_features.append(x)

        cur_fpn = fpns[2]
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # dcn for the last layer
        offset = self.conv_offset(x)
        x = self.dcn(x, offset)
        x = self.gn5(x)
        x = F.relu(x)
        multi_scale_features.append(x)
        return multi_scale_features


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights


class Transformer(nn.Module):

    def __init__(self, num_frames, backbone_dims, d_model=384, nhead=8,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 num_encoder_layers=(6,),
                 num_decoder_layers=6,
                 num_decoder_queries=1,
                 return_intermediate_dec=False,
                 bbox_nhead=8,
                 encoder_cross_layer=False,
                 use_soft_mask_encoder=False,
                 decoder_type='multiscale_query'):

        super().__init__()
        # import ipdb; ipdb.set_trace()
        self.decoder_type = decoder_type

        if not type(num_encoder_layers) in [list, tuple]:
            num_encoder_layers = [num_encoder_layers]
        self.num_encoder_layers = [num_encoder_layer if type(num_encoder_layer) is int else int(num_encoder_layer) for
                                   num_encoder_layer in num_encoder_layers]
        self.num_decoder_queries = num_decoder_queries
        self.backbone_dims = backbone_dims
        self.num_backbone_feats = len(backbone_dims)
        self.num_frames = num_frames
        self.input_proj_modules = nn.ModuleList()
        self.fpn = MaskHeadSmallConv(dim=d_model, context_dim=d_model)
        self.use_encoder = sum(self.num_encoder_layers) > 0
        self.num_encoder_stages = len(self.num_encoder_layers)
        self.use_decoder = num_decoder_layers > 0
        self.use_soft_mask_encoder = use_soft_mask_encoder

        if self.num_encoder_stages == 1 and encoder_cross_layer:
            self.num_encoder_stages = 2
            num_encoder_layers += (1,)
            self.num_encoder_layers = num_encoder_layers

        for backbone_dim in backbone_dims:
            self.input_proj_modules.append(nn.Conv2d(backbone_dim, d_model, kernel_size=1))
        if sum(self.num_encoder_layers) > 0:
            self.encoder = TransformerEncoder(self.num_encoder_layers, nhead, dim_feedforward, d_model, dropout,
                                              activation,
                                              normalize_before, use_cross_layers=encoder_cross_layer,
                                              cross_pos='cascade',
                                              use_soft_mask=use_soft_mask_encoder)

        if num_decoder_layers > 0:
            self.query_embed = nn.Embedding(num_decoder_queries, d_model)
            if 'adaptive' in self.decoder_type:
                decoder_layer = AdaptiveTransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                        dropout, activation, normalize_before)
                self.decoder_type = self.decoder_type.replace('_adaptive', '')
            else:
                decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                        dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(d_model)

            if self.decoder_type == 'multiscale_query':
                self.decoder = MultiscaleQueryTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                                                 return_intermediate=return_intermediate_dec)
            elif self.decoder_type == 'multiscale_memory':
                self.decoder = MultiscaleMemoryTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
            elif self.decoder_type == 'multiscale_query_memory':
                ######### Hard coding number of layers + 3 since we use an additional scale
                self.decoder =  nn.ModuleList([
                        MultiscaleQueryTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                                          return_intermediate=return_intermediate_dec),
                        MultiscaleMemoryTransformerDecoder(decoder_layer, num_decoder_layers-1, decoder_norm)])
            elif self.decoder_type == 'multiscale_query_memory_eff':
                ######### Hard coding number of layers + 3 since we use an additional scale
                self.decoder =  nn.ModuleList([
                        MultiscaleQueryTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                                          return_intermediate=return_intermediate_dec),
                        MultiscaleMemoryTransformerDecoder(decoder_layer, num_decoder_layers-3, decoder_norm, eff=True)])
            elif self.decoder_type == 'multiscale_query_memory_nobidir':
                ######### Hard coding number of layers - 1 since we use an additional scale
                self.decoder =  nn.ModuleList([
                        MultiscaleQueryTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                                          return_intermediate=return_intermediate_dec),
                        MultiscaleMemoryNoBidirTransformerDecoder(decoder_layer, num_decoder_layers-2, decoder_norm)])

            self.bbox_attention = MHAttentionMap(d_model, d_model, bbox_nhead, dropout=0.0)
        self.d_model = d_model
        self.nhead = nhead
        self.tgt_clip_tag = None
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, pos_list, batch_size, clip_tag=None, pred_mems=None):
        # features as TCHW
        # import ipdb; ipdb.set_trace()
        bt = features[-1].tensors.shape[0]
        bs_f = bt // self.num_frames
        # project all backbone features to transformer dim
        for i in range(len(features)):
            src, mask = features[i].decompose()
            assert mask is not None
            src_proj = self.input_proj_modules[i](src)
            features[i] = NestedTensor(src_proj, mask)

        # reshape all features to sequences for encoder
        if self.use_encoder:
            enc_feat_list = []
            enc_mask_list = []
            enc_pos_embd_list = []
            enc_feat_shapes = []
            pred_mems_flat = []
            for i in range(self.num_encoder_stages):
                feat_index = -1 - i  # taking in reverse order from deeper to shallow
                src_proj, mask = features[feat_index].decompose()
                assert mask is not None
                n, c, s_h, s_w = src_proj.shape
                enc_feat_shapes.append((s_h, s_w))
                src = src_proj.reshape(bs_f, self.num_frames, c, s_h, s_w).permute(0, 2, 1, 3, 4).flatten(-2)
                # bs, c, t, hw = src.shape
                src = src.flatten(2).permute(2, 0, 1)
                enc_feat_list.append(src)
                mask = mask.reshape(bs_f, self.num_frames, s_h * s_w)
                mask = mask.flatten(1)
                enc_mask_list.append(mask)
                pos_embed = pos_list[feat_index].permute(0, 2, 1, 3, 4).flatten(-2)
                pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
                enc_pos_embd_list.append(pos_embed)
                # import ipdb; ipdb.set_trace()
                if pred_mems is not None and len(pred_mems) > 0:
                    pred_mem_fi = pred_mems[feat_index]  # check with mask
                    pred_mem_fi = pred_mem_fi.reshape(bs_f, self.num_frames, s_h * s_w)
                    pred_mem_fi = pred_mem_fi.flatten(1)
                    pred_mems_flat.append(pred_mem_fi)
                # import ipdb;ipdb.set_trace()
                # print('inside enc_features>> i:%d feat_index:%d src.shape:%s mask.shape:%s pos.shape:%s' % (
                # i, feat_index, str(src.shape), str(mask.shape), str(pos_embed.shape)))
            encoder_features = self.encoder(features=enc_feat_list, src_key_padding_masks=enc_mask_list,
                                            pos_embeds=enc_pos_embd_list, sizes=enc_feat_shapes,
                                            pred_mems=pred_mems_flat)
            for i in range(self.num_encoder_stages):
                memory_i = encoder_features[i]
                h, w = enc_feat_shapes[i]
                memory_i = memory_i.permute(1, 2, 0).view(bs_f, self.d_model, self.num_frames, h * w)
                memory_i = memory_i.permute(0, 2, 1, 3).reshape(bs_f, self.num_frames, self.d_model, h, w).flatten(0, 1)
                encoder_features[i] = memory_i
                # print('enc output>> i:%d  memory_i.shape:%s' % (i, memory_i.shape))
            deep_feature = encoder_features[0]
            fpn_features = []
            for i in range(1, self.num_encoder_stages):
                fpn_features.append(encoder_features[i])
            for i in reversed(range(self.num_backbone_feats - self.num_encoder_stages)):
                _, c_f, h, w = features[i].tensors.shape
                features[i].tensors = features[i].tensors.reshape(bs_f, self.num_frames, c_f, h, w)
                fpn_features.append(features[i].tensors.flatten(0, 1))
        else:
            # print('Not using encoder>>> Not implemented yet as not doing experiment now')
            # import ipdb;ipdb.set_trace()
            # TODO check, not tested
            _, c_f, h, w = features[-1].tensors.shape
            features[-1].tensors = features[-1].tensors.reshape(bs_f, self.num_frames, c_f, h, w)
            deep_feature = features[-1].tensors.flatten(0, 1)
            fpn_features = []
            for i in reversed(range(self.num_backbone_feats - 1)):
                _, c_f, h, w = features[i].tensors.shape
                features[i].tensors = features[i].tensors.reshape(bs_f, self.num_frames, c_f, h, w)
                fpn_features.append(features[i].tensors.flatten(0, 1))
        ################################################################
        ###################################################################
        # import ipdb; ipdb.set_trace()
        ms_feats = self.fpn(deep_feature, fpn_features)
        hr_feat = ms_feats[-1]
        if self.use_decoder:
            dec_features = []
            pos_embed_list = []
            size_list = []
            dec_mask_list = []

            if self.decoder_type in ['multiscale_memory', 'multiscale_query_memory', 'multiscale_query_memory_eff',
                                     'multiscale_query_memory_nobidir', 'multiscale_query_memory_adaptive']:
                mscales = 4
                ms_feats.append(hr_feat)
            else:
                mscales = 3

            for i in range(mscales):
                fi = ms_feats[i]
                ni, ci, hi, wi = fi.shape
                fi = fi.reshape(bs_f, self.num_frames, ci, hi, wi).permute(
                        0, 2, 1, 3, 4).flatten(-2).flatten(2).permute(2, 0, 1)

                dec_mask_i = features[-1 - i].mask.reshape(bs_f, self.num_frames, hi * wi).flatten(1)
                pe = pos_list[-1 - i].permute(0, 2, 1, 3, 4).flatten(-2).flatten(2).permute(2, 0, 1)
                dec_features.append(fi)
                pos_embed_list.append(pe)
                size_list.append((hi, wi))
                dec_mask_list.append(dec_mask_i)
            query_embed = self.query_embed.weight
            query_embed = query_embed.unsqueeze(1)
            tq, bq, cq = query_embed.shape
            query_embed = query_embed.repeat(self.num_frames // tq, bs_f, 1)
            tgt = torch.zeros_like(query_embed)

            if self.decoder_type == 'multiscale_query':
                hs = self.decoder(tgt, dec_features, memory_key_padding_mask=dec_mask_list,
                                  pos=pos_embed_list, query_pos=query_embed, size_list=size_list)
            elif self.decoder_type == 'multiscale_memory':
                raise NotImplementedError()
            elif self.decoder_type in ['multiscale_query_memory', 'multiscale_query_memory_eff', \
                                       'multiscale_query_memory_nobidir', 'multiscale_query_memory_adaptive']:
                hs = self.decoder[0](tgt, dec_features[:mscales-1],
                                     memory_key_padding_mask=dec_mask_list[:mscales-1],
                                     pos=pos_embed_list[:mscales-1], query_pos=query_embed,
                                     size_list=size_list[:mscales-1])
                hr_feat = self.decoder[1](hs[-1], dec_features, feat_mask=dec_mask_list,
                                          feat_pos=pos_embed_list, memory_pos=query_embed,
                                          size_list=size_list, num_frames=self.num_frames)
                thw, _, c = hr_feat.shape
                hr_feat = hr_feat.permute(1, 2, 0).reshape(1, c, self.num_frames, thw//self.num_frames).reshape(
                                1, c, self.num_frames, size_list[3][0], size_list[3][1]).flatten(0,1).permute(1,0,2,3)

            hs = hs.transpose(1, 2)
            n_f = 1 if self.num_decoder_queries == 1 else self.num_decoder_queries // self.num_frames
            obj_attn_masks = []
            for i in range(self.num_frames):
                t2, c2, h2, w2 = hr_feat.shape
                hs_f = hs[-1][:, i * n_f:(i + 1) * n_f, :]
                memory_f = hr_feat[i, :, :, :].reshape(batch_size, c2, h2, w2)
                mask_f = features[0].mask[i, :, :].reshape(batch_size, h2, w2)
                obj_attn_mask_f = self.bbox_attention(hs_f, memory_f, mask=mask_f).flatten(0, 1)
                obj_attn_masks.append(obj_attn_mask_f)
            obj_attn_masks = torch.cat(obj_attn_masks, dim=0)
            seg_feats = torch.cat([hr_feat, obj_attn_masks], dim=1)
            # import ipdb; ipdb.set_trace()
            if clip_tag is not None:
                self.tgt_clip_tag = clip_tag
                self.tgt = hs[-1].permute(1, 0, 2)
        else:
            seg_feats = hr_feat
        return seg_feats


class TransformerEncoder(nn.Module):

    def __init__(self, num_encoder_layers, nhead, dim_feedforward, d_model, dropout, activation, normalize_before,
                 use_cross_layers, cross_pos, use_soft_mask):
        super().__init__()
        # ######################################
        # import ipdb; ipdb.set_trace()
        self.cross_pos = cross_pos  # pre, post, cascade
        self.num_layers = num_encoder_layers
        self.num_stages = len(num_encoder_layers)
        self.layers = nn.ModuleList()

        for i in range(len(num_encoder_layers)):
            # print('Encoder stage:%d dim_feedforward:%d' % (i, dim_feedforward))
            self.layers.append(nn.ModuleList())
            for j in range(num_encoder_layers[i]):
                _nhead = nhead if i == 0 else 1
                _dim_feedforward = dim_feedforward if i == 0 else d_model
                _dropout = dropout if i == 0 else 0
                _use_layer_norm = (i == 0)
                _use_soft_mask = i >= 0 and j < 6 and use_soft_mask  # use soft-mask only in first few layers of each stage, use param config later
                print('i:%d j:%d soft_mask:%s' % (i, j, str(_use_soft_mask)))
                encoder_layer = TransformerEncoderLayer(d_model, _nhead, _dim_feedforward, _dropout, activation,
                                                        normalize_before, _use_layer_norm, _use_soft_mask)
                self.layers[i].append(encoder_layer)
        # ######################################
        # import ipdb; ipdb.set_trace()
        # #############################################
        self.norm = nn.LayerNorm(d_model) if normalize_before else None
        self.cross_pos = cross_pos
        if self.num_stages > 1:
            self.norms = None if not normalize_before else nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(self.num_stages - 1)])
        self.use_cross_layers = use_cross_layers
        if use_cross_layers:
            self.cross_res_layers = nn.ModuleList([
                CrossResolutionEncoderLayer(d_model=384, nhead=1, dim_feedforward=384, layer_norm=False,
                                            custom_instance_norm=True, fuse='linear', soft_mask=use_soft_mask)
            ])

    def forward(self, features, src_key_padding_masks, pos_embeds, sizes, masks=None, pred_mems=None):
        outputs = []
        # import ipdb; ipdb.set_trace()
        for i in range(self.num_stages):
            output = features[i]
            skp_mask = src_key_padding_masks[i]
            pos_embed = pos_embeds[i]
            pred_mem_i = None
            if pred_mems is not None and len(pred_mems) > 0:
                pred_mem_i = pred_mems[i]
            if self.use_cross_layers and i > 0 and self.cross_pos == 'cascade':
                src0 = outputs[i - 1]
                src1 = output
                mask0 = src_key_padding_masks[i - 1]
                pos0 = pos_embeds[i - 1]
                mask1 = src_key_padding_masks[i]
                pos1 = pos_embeds[i]
                # import ipdb;ipdb.set_trace()
                output = self.cross_res_layers[0](src0, src1, mask0, mask1, pos0, pos1)
            # shape = sizes[i]
            # import ipdb; ipdb.set_trace()
            for j in range(self.num_layers[i]):
                output = self.layers[i][j](output, src_mask=None, src_key_padding_mask=skp_mask, pos=pos_embed,
                                           pred_mem=pred_mem_i)
            outputs.append(output)
        if self.use_cross_layers and self.cross_pos == 'post':
            src0 = outputs[0]
            if self.num_stages > 1:
                src1 = outputs[1]
            else:
                src1 = features[1]
                outputs.append(src1)
            mask0 = src_key_padding_masks[0]
            pos0 = pos_embeds[0]
            mask1 = src_key_padding_masks[1]
            pos1 = pos_embeds[1]
            # import ipdb;ipdb.set_trace()
            output1 = self.cross_res_layers[0](src0, src1, mask0, mask1, pos0, pos1)
            # import ipdb;ipdb.set_trace()
            outputs[1] = output1
        # TODO
        # Check this norm later
        if self.norm is not None:
            outputs[0] = self.norm(outputs[0])
            if self.num_stages > 1:
                for i in range(self.num_stages - 1):
                    outputs[i + 1] = self.norms[i](outputs[i + 1])
        return outputs

class MultiscaleMemoryTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, eff=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        if eff:
            self.iterate_levels = [0, 1, 2, 1, 2, 3]
        else:
            self.iterate_levels = [0, 1, 2, 1, 0, 1, 2, 3]

    def _mix_levels(self, low_res_f, high_res_f, size_low, size_high, num_frames):
        low_res_f = low_res_f.view(num_frames, *size_low, *low_res_f.shape[-2:])
        low_res_f = low_res_f.permute(3, 0, 4, 1, 2)

        bs, t, c, h, w = low_res_f.shape
        low_res_f = F.interpolate(low_res_f.flatten(0, 1), size_high, mode='bilinear')

        low_res_f = low_res_f.view(bs, t, c, *size_high).permute(1, 3, 4, 0, 2)
        mixed_feats = high_res_f + low_res_f.flatten(0,2)
        return mixed_feats

    def forward(self, memory, dec_features,
                memory_mask: Optional[Tensor] = None,
                feat_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                feat_key_padding_mask: Optional[Tensor] = None,
                feat_pos: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,
                size_list=None, num_frames=-1):

        device = memory.device

        output = None
        intermediate = []

        for it, feat_index in enumerate(self.iterate_levels):
#            print('==============>', it, ' : ', feat_index)
#            a = torch.cuda.memory_allocated(device)
            if output is None:
                output = dec_features[feat_index]
            else:
                # Mix between attended features w.r.t slots + higher level feats
                feat_index_bar = self.iterate_levels[it-1]
                output = self._mix_levels(output, dec_features[feat_index], size_list[feat_index_bar],
                                          size_list[feat_index], num_frames)
#            b = torch.cuda.memory_allocated(device)
#            print('Memory from interpolation ', (b - a)/1000000.0)

            pos_i = feat_pos[feat_index]
            feat_mask_i = feat_mask[feat_index]
            output = self.layers[it](output, memory, tgt_mask=None,
                           memory_mask=None,
                           tgt_key_padding_mask=feat_mask_i,
                           memory_key_padding_mask=None,
                           pos=memory_pos, query_pos=pos_i, size_list=size_list)
#            c = torch.cuda.memory_allocated(device)
#            print('Memory from attention ', (c-b)/1000000.0)

        if self.norm is not None:
            output = self.norm(output)
            #if self.return_intermediate:
            #    intermediate.pop()
            #    intermediate.append(dec_features[-1])
        #if self.return_intermediate:
        #    return torch.stack(intermediate)
        return output

class MultiscaleMemoryNoBidirTransformerDecoder(MultiscaleMemoryTransformerDecoder):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, eff=False):
        super(MultiscaleMemoryNoBidirTransformerDecoder, self).__init__(
                decoder_layer, num_layers, norm=norm, return_intermediate=return_intermediate, eff=eff)

        self.iterate_levels = [0, 1, 2, 0, 1, 2, 3]

class MultiscaleQueryTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, dec_features,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                size_list=None):
        output = tgt
        intermediate = []
        for idx, layer in enumerate(self.layers):
            feat_index = idx % (len(dec_features))
            current_dec_feat = dec_features[feat_index]
            pos_i = pos[feat_index]
            memory_key_padding_mask_i = memory_key_padding_mask[feat_index]
            output = layer(output, current_dec_feat, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask_i,
                           pos=pos_i, query_pos=query_pos, size_list=size_list)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output


class CrossResolutionEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, layer_norm, custom_instance_norm, fuse, soft_mask,
                 dropout=0, activation="relu"):
        super().__init__()
        print('Creating cross resolution layer>>> d_model: %3d nhead:%d dim_feedforward:%3d' % (
            d_model, nhead, dim_feedforward))
        assert fuse is None or fuse in ['simple', 'linear', 'bilinear']
        self.fuse = fuse  # None, 'simple, ''', 'linear', 'bilinear'
        self.use_soft_mask = soft_mask
        self.nhead = nhead
        if self.fuse is None:
            pass
        elif self.fuse == 'simple':
            pass
        elif self.fuse == 'linear':
            self.fuse_linear1 = nn.Linear(d_model, d_model)
            self.fuse_linear2 = nn.Linear(d_model, d_model)
            self.fuse_linear3 = nn.Linear(d_model, d_model)
        elif self.fuse == 'bilinear':
            self.bilinear = nn.Bilinear(d_model, d_model, d_model)
        else:
            raise ValueError('fuse:{} not recognized'.format(self.fuse))
        if self.use_soft_mask:
            self.soft_mask_sigma = nn.Parameter(torch.ones(nhead) * 1e-3)
            self.soft_mask_alpha = nn.Parameter(nn.Parameter(torch.ones(nhead) * 1e-2))
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.custom_instance_norm = custom_instance_norm
        self.norm1 = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.res_embed = nn.Embedding(1, d_model)
        self.res_embed_shallow = nn.Embedding(1, d_model)
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_deep, feat_shallow, mask_deep, mask_shallow, pos_deep, pos_shallow):
        # import ipdb; ipdb.set_trace()
        if self.fuse is None:
            _f_deep = feat_deep
        elif self.fuse == 'simple':
            _f_deep = feat_deep + feat_shallow
        elif self.fuse == 'linear':
            _f_deep = self.fuse_linear3(
                self.activation(self.fuse_linear1(feat_deep)) + self.activation(self.fuse_linear2(feat_shallow)))
        elif self.fuse == 'bilinear':
            _f_deep = self.bilinear(feat_deep, feat_shallow)
        else:
            raise ValueError('fuse:{} not recognized'.format(self.fuse))
        res_embed = self.res_embed.weight.unsqueeze(0).repeat(pos_deep.shape[0], 1, 1)
        res_embed_shallow = self.res_embed_shallow.weight.unsqueeze(0).repeat(pos_deep.shape[0], 1, 1)
        if self.custom_instance_norm:
            deep_u = feat_deep.mean(dim=0)
            deep_s = feat_deep.std(dim=0)
            shallow_u = feat_shallow.mean(dim=0)
            shallow_s = feat_shallow.std(dim=0)
            _f_deep = (_f_deep - deep_u)
            if deep_s.min() > 1e-10:
                _f_deep = _f_deep / deep_s
            _f_deep = (_f_deep * shallow_s) + shallow_u
        _f_deep = _f_deep * self.gamma + self.beta
        kp = _f_deep + pos_deep + res_embed
        qp = feat_shallow + pos_shallow + res_embed_shallow
        vv = feat_shallow
        # import ipdb; ipdb.set_trace()
        attn_mask = torch.mm(mask_shallow.transpose(1, 0).double(), mask_deep.double()).bool()
        if self.use_soft_mask:
            # import ipdb; ipdb.set_trace()
            batches = feat_shallow.shape[1]
            seq_len = feat_shallow.shape[0]
            x = torch.linspace(0.0, 1.0, steps=seq_len)
            y = torch.linspace(0.0, 1.0, steps=seq_len)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            dist_mat = (xx - yy).to(self.soft_mask_sigma.device)
            dist_mat = torch.pow(dist_mat, 2)
            dist_mat = dist_mat.unsqueeze(0).repeat(self.nhead, 1, 1)
            sigma_square = torch.pow(self.soft_mask_sigma, 2).unsqueeze(1).unsqueeze(1)
            alpha_square = torch.pow(self.soft_mask_alpha, 2).unsqueeze(1).unsqueeze(1)
            attn_mask2 = alpha_square * torch.exp(-dist_mat / (2 * sigma_square + 1e-6))
            attn_mask2 = attn_mask2.repeat(batches, 1, 1)
            # attn_mask = attn_mask.unsqueeze(0).repeat(self.nhead* batches, 1, 1)
            attn_mask = attn_mask2
        out = self.self_attn(qp, kp, value=vv, attn_mask=attn_mask, key_padding_mask=mask_shallow)[0]
        out = feat_shallow + self.dropout1(out)
        out = self.norm1(out)
        out2 = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = out + self.dropout2(out2)
        out = self.norm2(out)
        return out


class CrossResolutionEncoderLayerBack(nn.Module):

    def __init__(self, d_model, nhead=1, dim_feedforward=384, dropout=0, activation="relu", custom_instance_norm=True,
                 layer_norm=False, bilinear_layer=True):
        super().__init__()
        print('Creating cross resolution layer>>> d_model: %3d nhead:%d dim_feedforward:%3d' % (
            d_model, nhead, dim_feedforward))
        self.use_bilinear_layer = bilinear_layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.custom_instance_norm = custom_instance_norm
        self.norm1 = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.res_embed = nn.Embedding(1, d_model)
        self.res_embed_shallow = nn.Embedding(1, d_model)
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_deep, feat_shallow, mask_deep, mask_shallow, pos_deep, pos_shallow):
        _f_deep = feat_deep
        res_embed = self.res_embed.weight.unsqueeze(0).repeat(pos_deep.shape[0], 1, 1)
        res_embed_shallow = self.res_embed_shallow.weight.unsqueeze(0).repeat(pos_deep.shape[0], 1, 1)
        if self.custom_instance_norm:
            deep_u = feat_deep.mean(dim=0)
            deep_s = feat_deep.std(dim=0)
            shallow_u = feat_shallow.mean(dim=0)
            shallow_s = feat_shallow.std(dim=0)
            _f_deep = (_f_deep - deep_u)
            if deep_s.min() > 1e-10:
                _f_deep = _f_deep / deep_s
            _f_deep = (_f_deep * shallow_s) + shallow_u
            _f_deep = _f_deep * self.gamma + self.beta
        kp = _f_deep + pos_deep + res_embed
        qp = feat_shallow + pos_shallow + res_embed_shallow
        attn_mask = torch.mm(mask_shallow.transpose(1, 0).double(), mask_deep.double()).bool()
        out = self.self_attn(qp, kp, value=feat_shallow, attn_mask=attn_mask, key_padding_mask=mask_shallow)[0]
        out = feat_shallow + self.dropout1(out)
        out = self.norm1(out)
        out2 = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = out + self.dropout2(out2)
        out = self.norm2(out)
        return out


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, use_layer_norm=True, use_soft_mask=False):
        super().__init__()
        self.nhead = nhead
        self.use_layer_norm = use_layer_norm
        self.use_soft_mask = use_soft_mask
        if self.use_soft_mask:
            self.soft_mask_sigma = nn.Parameter(torch.ones(nhead) * 1e-4)
            self.soft_mask_alpha = nn.Parameter(torch.ones(nhead) * 1e-2)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     pred_mem: Tensor = None):
        # import ipdb; ipdb.set_trace()
        q = k = self.with_pos_embed(src, pos)
        self_attn_mask = src_mask
        if self.use_soft_mask:
            # import ipdb; ipdb.set_trace()
            batches = src.shape[1]
            seq_len = src.shape[0]
            x = torch.linspace(0.0, 1.0, steps=seq_len)
            y = torch.linspace(0.0, 1.0, steps=seq_len)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            dist_mat = (xx - yy).to(self.soft_mask_sigma.device)
            dist_mat = torch.pow(dist_mat, 2)
            dist_mat = dist_mat.unsqueeze(0).repeat(self.nhead, 1, 1)
            sigma_square = torch.pow(self.soft_mask_sigma, 2).unsqueeze(1).unsqueeze(1)
            alpha_square = torch.pow(self.soft_mask_alpha, 2).unsqueeze(1).unsqueeze(1)
            attn_mask2 = alpha_square * torch.exp(-dist_mat / (2 * sigma_square + 1e-7))
            attn_mask2 = attn_mask2.repeat(batches, 1, 1)
            self_attn_mask = attn_mask2
            # import ipdb; ipdb.set_trace()
        if pred_mem is not None:
            # #########
            history_mask = torch.mm(pred_mem.transpose(1, 0).double(), pred_mem.double())
            if self_attn_mask is not None:
                self_attn_mask = history_mask * self_attn_mask.double()
            else:
                self_attn_mask = history_mask
        src2 = self.self_attn(q, k, value=src, attn_mask=self_attn_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # import ipdb; ipdb.set_trace()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # import ipdb; ipdb.sset_trace()
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                pred_mem: Tensor = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, pred_mem=pred_mem)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     size_list=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        query_ca = self.with_pos_embed(tgt, query_pos)
        key_ca = self.with_pos_embed(memory, pos)
        value_ca = memory
        tgt2 = self.multihead_attn(query=query_ca,
                                   key=key_ca,
                                   value=value_ca, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                size_list=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, size_list=size_list)

class AdaptiveTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):

        super(AdaptiveTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout,
                                                              activation, normalize_before)

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     size_list=None):

        adaptive = False
        if tgt.shape[0] > memory.shape[0]:
            if (memory.shape[0] * size_list[-1][0] * size_list[-1][1] == tgt.shape[0]) or \
                    (memory.shape[0] * size_list[-2][0] * size_list[-2][1] == tgt.shape[0]):
                # Only enable the adaptive part in the two finest scale and when performing multiscale memory not query
                adaptive = True

        q = k = self.with_pos_embed(tgt, query_pos)
        if adaptive:
            N = int(q.shape[0] * 0.5)
            selected_indices = torch.randperm(q.shape[0])[:N].to(q.device)
            q, k, tgt_tmp, tgt_tmp_mask = q[selected_indices], k[selected_indices], tgt[selected_indices], \
                                            tgt_key_padding_mask[:, selected_indices]

            tgt2 = self.self_attn(q, k, value=tgt_tmp, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_tmp_mask)[0]
            tgt2_tmp = tgt.clone()
            tgt2_tmp[selected_indices] = tgt2
            tgt2 = tgt2_tmp

        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        query_ca = self.with_pos_embed(tgt, query_pos)
        key_ca = self.with_pos_embed(memory, pos)
        value_ca = memory
        tgt2 = self.multihead_attn(query=query_ca,
                                   key=key_ca,
                                   value=value_ca, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class MaskConsistencyDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(1, 1, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, vdim=1, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(1)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     size_list=None):
        import ipdb;
        ipdb.set_trace()
        # tgt is mask here
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        query_ca = key_ca = self.with_pos_embed(memory, pos)
        value_ca = tgt
        import ipdb;
        ipdb.set_trace()

        tgt2 = self.multihead_attn(query=query_ca,
                                   key=key_ca,
                                   value=value_ca, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                size_list=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, size_list=size_list)


class VisTR(nn.Module):

    def __init__(self, backbone, transformer, num_classes, num_frames, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = transformer.d_model
        self.num_frames = num_frames
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.input_proj_2 = nn.Conv2d(backbone.num_channels // 2, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, samples: NestedTensor):
        pass


class MEDVT(nn.Module):
    def __init__(self, vistr, freeze_vistr=False):
        super().__init__()
        self.vistr = vistr
        if freeze_vistr:
            for p in self.parameters():
                p.requires_grad_(False)
        hidden_dim, nheads = vistr.transformer.d_model, vistr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.insmask_head = nn.Sequential(
            nn.Conv3d(392, 384, (1, 3, 3), padding='same', dilation=1),
            nn.GroupNorm(4, 384),
            nn.ReLU(),
            nn.Conv3d(384, 256, 3, padding='same', dilation=2),
            nn.GroupNorm(4, 256),
            nn.ReLU(),
            nn.Conv3d(256, 128, 3, padding='same', dilation=2),
            nn.GroupNorm(4, 128),
            nn.ReLU(),
            nn.Conv3d(128, 1, 1))

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos_list = self.vistr.backbone(samples)  # ## check warnings on floor_divide
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()
        assert mask is not None
        src_proj = self.vistr.input_proj(src)
        n, c, s_h, s_w = src_proj.shape
        bs_f = bs // self.vistr.num_frames
        src_proj = src_proj.reshape(bs_f, self.vistr.num_frames, c, s_h, s_w).permute(0, 2, 1, 3, 4).flatten(-2)
        mask = mask.reshape(bs_f, self.vistr.num_frames, s_h * s_w)
        pos = pos_list[-1].permute(0, 2, 1, 3, 4).flatten(-2)
        src_2, mask_2 = features[-2].decompose()
        assert mask_2 is not None
        src_proj_2 = self.vistr.input_proj_2(src_2)
        n2, c2, s_h2, s_w2 = src_proj_2.shape
        src_proj_2 = src_proj_2.reshape(bs_f, self.vistr.num_frames, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
        mask_2 = mask_2.reshape(bs_f, self.vistr.num_frames, s_h2 * s_w2)
        pos_2 = pos_list[-2].permute(0, 2, 1, 3, 4).flatten(-2)
        hs, hr_feat = self.vistr.transformer(s_h, s_w, s_h2, s_w2, self.vistr.num_frames, src_proj, mask,
                                             self.vistr.query_embed.weight, pos, src_proj_2, mask_2, pos_2, features,
                                             pos_list)
        n_f = 1 if self.vistr.num_queries == 1 else self.vistr.num_queries // self.vistr.num_frames
        obj_attn_masks = []
        for i in range(self.vistr.num_frames):
            t2, c2, h2, w2 = hr_feat.shape
            hs_f = hs[-1][:, i * n_f:(i + 1) * n_f, :]
            memory_f = hr_feat[i, :, :, :].reshape(bs_f, c2, h2, w2)
            mask_f = features[0].mask[i, :, :].reshape(bs_f, h2, w2)
            obj_attn_mask_f = self.bbox_attention(hs_f, memory_f, mask=mask_f).flatten(0, 1)
            obj_attn_masks.append(obj_attn_mask_f)

        obj_attn_masks = torch.cat(obj_attn_masks, dim=0)
        seg_feats = torch.cat([hr_feat, obj_attn_masks], dim=1)
        mask_ins = seg_feats.unsqueeze(0).permute(0, 2, 1, 3, 4)
        outputs_seg_masks = self.insmask_head(mask_ins)
        outputs_seg_masks = outputs_seg_masks.squeeze(0)
        out = {"pred_masks": outputs_seg_masks}
        return out


class VOS_MEDVT(nn.Module):
    def __init__(self, args, backbone, backbone_dims, hidden_dim):
        super().__init__()
        self.backbone_name = args.backbone
        self.backbone = backbone
        if args.backbone == 'resnet101' or args.backbone == 'resnet50':
            self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        else:
            self.input_proj = nn.Conv3d(backbone_dims[-1], hidden_dim, kernel_size=1)
        self.insmask_head = nn.Sequential(
            nn.Conv3d(384, 384, (3, 3, 3), padding='same', dilation=1),
            nn.GroupNorm(4, 384),
            nn.ReLU(),
            nn.Conv3d(384, 256, 3, padding='same', dilation=2),
            nn.GroupNorm(4, 256),
            nn.ReLU(),
            nn.Conv3d(256, 128, 3, padding='same', dilation=2),
            nn.GroupNorm(4, 128),
            nn.ReLU(),
            nn.Conv3d(128, 1, 1))

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        if self.backbone_name == 'resnet101' or self.backbone_name == 'resnet50':
            features, pos_list = self.backbone(samples)  # ## check warnings on floor_divide
            src, mask = features[-1].decompose()
            src_proj = self.input_proj(src)
            src_proj = src_proj.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            features = self.backbone(samples.tensors.permute(1, 0, 2, 3).unsqueeze(0))
            src_proj = self.input_proj(features)

        """
        features, pos_list = self.vistr.backbone(samples)  # ## check warnings on floor_divide
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()
        assert mask is not None
        src_proj = self.vistr.input_proj(src)
        n, c, s_h, s_w = src_proj.shape
        bs_f = bs // self.vistr.num_frames
        src_proj = src_proj.reshape(bs_f, self.vistr.num_frames, c, s_h, s_w).permute(0, 2, 1, 3, 4).flatten(-2)
        mask = mask.reshape(bs_f, self.vistr.num_frames, s_h * s_w)
        pos = pos_list[-1].permute(0, 2, 1, 3, 4).flatten(-2)
        src_2, mask_2 = features[-2].decompose()
        assert mask_2 is not None
        src_proj_2 = self.vistr.input_proj_2(src_2)
        n2, c2, s_h2, s_w2 = src_proj_2.shape
        src_proj_2 = src_proj_2.reshape(bs_f, self.vistr.num_frames, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
        mask_2 = mask_2.reshape(bs_f, self.vistr.num_frames, s_h2 * s_w2)
        pos_2 = pos_list[-2].permute(0, 2, 1, 3, 4).flatten(-2)
        hs, hr_feat = self.vistr.transformer(s_h, s_w, s_h2, s_w2, self.vistr.num_frames, src_proj, mask,
                                             self.vistr.query_embed.weight, pos, src_proj_2, mask_2, pos_2, features,
                                             pos_list)
        n_f = 1 if self.vistr.num_queries == 1 else self.vistr.num_queries // self.vistr.num_frames
        obj_attn_masks = []
        for i in range(self.vistr.num_frames):
            t2, c2, h2, w2 = hr_feat.shape
            hs_f = hs[-1][:, i * n_f:(i + 1) * n_f, :]
            memory_f = hr_feat[i, :, :, :].reshape(bs_f, c2, h2, w2)
            mask_f = features[0].mask[i, :, :].reshape(bs_f, h2, w2)
            obj_attn_mask_f = self.bbox_attention(hs_f, memory_f, mask=mask_f).flatten(0, 1)
            obj_attn_masks.append(obj_attn_mask_f)
        obj_attn_masks = torch.cat(obj_attn_masks, dim=0)
        seg_feats = torch.cat([hr_feat, obj_attn_masks], dim=1)
        mask_ins = seg_feats.unsqueeze(0).permute(0, 2, 1, 3, 4)
        """
        mask_ins = src_proj
        outputs_seg_masks = self.insmask_head(mask_ins)
        outputs_seg_masks = outputs_seg_masks.squeeze(0)
        out = {"pred_masks": outputs_seg_masks}
        return out

class VOS_SwinMEDVT(nn.Module):
    def __init__(self, args, backbone, backbone_dims, hidden_dim, transformer, num_frames, temporal_strides=[1]):
        super().__init__()
        # import ipdb; ipdb.set_trace()
        self.use_memory_mask_enc = args.use_mem_mask
        self.backbone_name = args.backbone
        self.backbone = backbone
        self.num_frames = num_frames
        self.position_embedding = build_position_encoding(args)
        self.temporal_strides = temporal_strides
        self.transformer = transformer
        self.use_soft_mask_encoder = args.use_soft_mask_encoder
        self.use_mask_consistency_decoder = args.mask_consistency_decoder
        if self.use_memory_mask_enc:
            self.mask_mem = LRU(maxsize=5)
        if transformer is None:
            self.input_proj = nn.Conv2d(backbone_dims[-1], hidden_dim, kernel_size=1)
        if transformer is None or not transformer.use_decoder:
            self.insmask_head = nn.Sequential(
                nn.Conv3d(384, 384, (1, 3, 3), padding='same', dilation=1),
                nn.GroupNorm(4, 384),
                nn.ReLU(),
                nn.Conv3d(384, 256, 3, padding='same', dilation=2),
                nn.GroupNorm(4, 256),
                nn.ReLU(),
                nn.Conv3d(256, 128, 3, padding='same', dilation=2),
                nn.GroupNorm(4, 128),
                nn.ReLU(),
                nn.Conv3d(128, 1, 1))
        else:
            self.insmask_head = nn.Sequential(
                nn.Conv3d(392, 384, (1, 3, 3), padding='same', dilation=1),
                nn.GroupNorm(4, 384),
                nn.ReLU(),
                nn.Conv3d(384, 256, 3, padding='same', dilation=2),
                nn.GroupNorm(4, 256),
                nn.ReLU(),
                nn.Conv3d(256, 128, 3, padding='same', dilation=2),
                nn.GroupNorm(4, 128),
                nn.ReLU(),
                nn.Conv3d(128, 1, 1))
        if self.use_mask_consistency_decoder:
            self.mask_consistency_decoder_layer = MaskConsistencyDecoderLayer(d_model=384, nhead=1, dim_feedforward=1)

    def _divide_by_stride(self, samples: NestedTensor):
        samples_dict = {}
        for it, stride in enumerate(self.temporal_strides):
            start = it * self.num_frames
            end = (it + 1) * self.num_frames
            samples_dict[stride] = NestedTensor(samples.tensors[start:end], samples.mask[start:end])
        return samples_dict

    def forward(self, samples: NestedTensor, clip_tag=None):
        if self.training:
            return self._forward_one_samples(samples, clip_tag)
        else:
            return self.forward_inference(samples, clip_tag)

    def forward_inference(self, samples: NestedTensor, clip_tag=None):
        samples = self._divide_by_stride(samples)
        all_outs = []
        for stride, samples_ in samples.items():
            all_outs.append(self._forward_one_samples(samples_, clip_tag=clip_tag))
        all_outs = torch.stack([a['pred_masks'] for a in all_outs], dim=0)
        return {'pred_masks': all_outs.mean(0)}

    def _forward_one_samples(self, samples: NestedTensor, clip_tag=None):
        if not self.training and self.use_memory_mask_enc:
            if '@flipped' in clip_tag:
                clip_tag_flip = clip_tag
                clip_tag_no_flip = clip_tag.replace('@flipped', '')
            else:
                clip_tag_flip = '%s@flipped' % clip_tag
                clip_tag_no_flip = clip_tag
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        # import ipdb; ipdb.set_trace()
        if self.backbone_name == 'resnet101' or self.backbone_name == 'resnet50':
            features, pos_list = self.backbone(samples)  # ## check warnings on floor_divide
            #for i in range(1):
            #    features[i].tensors = F.interpolate(features[i].tensors, features[i+1].tensors.shape[-2:])
            #    features[i].mask = F.interpolate(features[i].mask.unsqueeze(1).float(),
            #                                     features[i+1].tensors.shape[-2:], mode='nearest').bool()
            #    pos_list[i] = copy.deepcopy(pos_list[i+1])
            batch_size = 1
        else:
            features = self.backbone(samples.tensors.permute(1, 0, 2, 3).unsqueeze(0))
            batch_size, _, num_frames, _, _ = features[0].shape
            pos_list = []
            for i in range(len(features)):
                x = features[i].permute(0, 2, 1, 3, 4).flatten(0, 1)
                m = samples.mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                x = NestedTensor(x, mask)
                features[i] = x
                pos_list.append(self.position_embedding(x).to(x.tensors.dtype))
        # import ipdb; ipdb.set_trace()
        if self.transformer is None:
            src, mask = features[-1].decompose()
            src_proj = self.input_proj(src)
            mask_ins = src_proj.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            pred_mems = []
            # import ipdb; ipdb.set_trace()
            if not self.training and self.use_memory_mask_enc:
                # import ipdb; ipdb.set_trace()
                if clip_tag_no_flip in self.mask_mem:
                    prev_mem = self.mask_mem[clip_tag_no_flip]
                    if clip_tag == clip_tag_flip:
                        prev_mem = prev_mem.flip(-1)
                    for i in range(len(features)):
                        pred_scale_i = F.interpolate(prev_mem, size=features[i].tensors.shape[-2:])
                        pred_mems.append(pred_scale_i.squeeze(0))
            seg_feats = self.transformer(features, pos_list, batch_size, clip_tag=clip_tag, pred_mems=pred_mems)
            mask_ins = seg_feats.unsqueeze(0).permute(0, 2, 1, 3, 4)
        outputs_seg_masks = self.insmask_head(mask_ins)
        if self.use_mask_consistency_decoder:
            # import ipdb; ipdb.set_trace()
            bt, _, m_t, m_h, m_w = outputs_seg_masks.shape
            bt, f_c, f_t, f_h, f_w = mask_ins.shape
            seq_feats_seqs = mask_ins[:, :384, :, :, :].flatten(-2).flatten(-2).permute(2, 0, 1)
            seg_masks_seqs = outputs_seg_masks.flatten(-2).flatten(-2).permute(2, 0, 1)
            bs_f = (bt * f_t) // self.num_frames
            mask = features[0].mask.reshape(bs_f, self.num_frames, f_h * f_w).flatten(1)
            pos_embed = pos_list[0].permute(0, 2, 1, 3, 4).flatten(-2).flatten(-2).permute(2, 0, 1)
            # import ipdb; ipdb.set_trace()
            refined_masks = self.mask_consistency_decoder_layer(tgt=seg_masks_seqs, memory=seq_feats_seqs,
                                                                pos=pos_embed, query_pos=None)

            refined_masks = refined_masks.permute(1, 2, 0).view(bs_f, self.d_model, self.num_frames, m_h * m_w)
            refined_masks = refined_masks.permute(0, 2, 1, 3).reshape(bs_f, self.num_frames, self.d_model, m_h,
                                                                      m_w).flatten(0, 1)
            outputs_seg_masks = refined_masks
        else:
            outputs_seg_masks = outputs_seg_masks.squeeze(0)
        out = {"pred_masks": outputs_seg_masks}
        # import ipdb;ipdb.set_trace()
        if not self.training and self.use_memory_mask_enc:
            # self.mask_mem.clear()
            # import ipdb;ipdb.set_trace()
            new_mem = torch.sigmoid(outputs_seg_masks.clone().detach())
            new_mem = torchvision.transforms.functional.gaussian_blur(new_mem, kernel_size=(7, 7))
            if clip_tag == clip_tag_flip:
                new_mem = new_mem.flip(-1)
            if clip_tag_no_flip in self.mask_mem:
                prev_mem = self.mask_mem[clip_tag_no_flip]
                # prev_shape = prev_mem.shape[2:]
                # new_shape = new_mem.shape[2:]
                save_shape = max(prev_mem.shape[-2:], new_mem.shape[-2:])
                if new_mem.shape[-2:] != save_shape:
                    new_mem = F.interpolate(new_mem, size=save_shape)
                if prev_mem.shape[-2:] != save_shape:
                    prev_mem = F.interpolate(prev_mem, size=save_shape)
                new_mem = 0.8 * new_mem + 0.2 * prev_mem
                # TODO naive approach, check and improvise this later
            self.mask_mem[clip_tag_no_flip] = new_mem
        return out

class VOS_SwinMEDVTLPROP(VOS_SwinMEDVT):
    def __init__(self, args, backbone, backbone_dims, hidden_dim, transformer, num_frames,
                 pretrain_settings={}, lprop_mode=None, temporal_strides=[1], feat_loc=None,
                 stacked=1):

        super().__init__(args, backbone, backbone_dims, hidden_dim, transformer, num_frames)
        self.stacked = stacked

        if feat_loc == 'early_coarse':
            feat_dim = 384
            hidden_dim = 128
        if feat_loc == 'early_fine':
            feat_dim = 384
            hidden_dim = 128
        elif feat_loc == 'late':
            feat_dim = 392
            hidden_dim = 128
        elif feat_loc == 'attmaps_only':
            feat_dim = 8
            hidden_dim = 16

        self.lprop_mode = lprop_mode
        self.label_propagator = LabelPropagator(lprop_mode, feat_dim=feat_dim, hidden_dim=hidden_dim)
        self.feat_loc = feat_loc
        self.temporal_strides = temporal_strides

        # Pretraining Settings
        if pretrain_settings is None:
            pretrain_settings = {}

        if 'freeze_pretrained' not in pretrain_settings:
            pretrain_settings['freeze_pretrained'] = False
            pretrain_settings['pretrained_model_path'] = ''

        if 'pretrain_label_enc' not in pretrain_settings:
            pretrain_settings['pretrain_label_enc'] = False
        else:
            if pretrain_settings['pretrain_label_enc']:
                assert 'label_enc_pretrain_path' in pretrain_settings, \
                        "Label encoder pretrained weights path needed"

        if pretrain_settings['freeze_pretrained']:
            self._freeze_pretrained_modules()

        if pretrain_settings['pretrain_label_enc']:
            checkpoint = torch.load(pretrain_settings['label_enc_pretrain_path'])
            self.label_propagator.label_encoder.load_state_dict(checkpoint, strict=False)

    def _freeze_pretrained_modules(self):
        pretrained_modules = [self.backbone, self.transformer]
        for mod in pretrained_modules:
            for p in mod.parameters():
                p._requires_grad = False
                p.requires_grad = False

    def _forward_one_samples(self, samples: NestedTensor, clip_tag=None):
        if not self.training and self.use_memory_mask_enc:
            if '@flipped' in clip_tag:
                clip_tag_flip = clip_tag
                clip_tag_no_flip = clip_tag.replace('@flipped', '')
            else:
                clip_tag_flip = '%s@flipped' % clip_tag
                clip_tag_no_flip = clip_tag
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        # import ipdb; ipdb.set_trace()
        if self.backbone_name == 'resnet101' or self.backbone_name == 'resnet50':
            features, pos_list = self.backbone(samples)  # ## check warnings on floor_divide
        else:
            features = self.backbone(samples.tensors.permute(1, 0, 2, 3).unsqueeze(0))
            batch_size, _, num_frames, _, _ = features[0].shape
            pos_list = []
            for i in range(len(features)):
                x = features[i].permute(0, 2, 1, 3, 4).flatten(0, 1)
                m = samples.mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                x = NestedTensor(x, mask)
                features[i] = x
                pos_list.append(self.position_embedding(x).to(x.tensors.dtype))
        # import ipdb; ipdb.set_trace()
        if self.transformer is None:
            src, mask = features[-1].decompose()
            src_proj = self.input_proj(src)
            mask_ins = src_proj.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            pred_mems = []
            # import ipdb; ipdb.set_trace()
            if not self.training and self.use_memory_mask_enc:
                # import ipdb; ipdb.set_trace()
                if clip_tag_no_flip in self.mask_mem:
                    prev_mem = self.mask_mem[clip_tag_no_flip]
                    if clip_tag == clip_tag_flip:
                        prev_mem = prev_mem.flip(-1)
                    for i in range(len(features)):
                        pred_scale_i = F.interpolate(prev_mem, size=features[i].tensors.shape[-2:])
                        pred_mems.append(pred_scale_i.squeeze(0))
            seg_feats = self.transformer(features, pos_list, batch_size, clip_tag=clip_tag, pred_mems=pred_mems)
            mask_ins = seg_feats.unsqueeze(0).permute(0, 2, 1, 3, 4)
        outputs_seg_masks = self.insmask_head(mask_ins)
        if self.use_mask_consistency_decoder:
            # import ipdb; ipdb.set_trace()
            bt, _, m_t, m_h, m_w = outputs_seg_masks.shape
            bt, f_c, f_t, f_h, f_w = mask_ins.shape
            seq_feats_seqs = mask_ins[:, :384, :, :, :].flatten(-2).flatten(-2).permute(2, 0, 1)
            seg_masks_seqs = outputs_seg_masks.flatten(-2).flatten(-2).permute(2, 0, 1)
            bs_f = (bt * f_t) // self.num_frames
            mask = features[0].mask.reshape(bs_f, self.num_frames, f_h * f_w).flatten(1)
            pos_embed = pos_list[0].permute(0, 2, 1, 3, 4).flatten(-2).flatten(-2).permute(2, 0, 1)
            # import ipdb; ipdb.set_trace()
            refined_masks = self.mask_consistency_decoder_layer(tgt=seg_masks_seqs, memory=seq_feats_seqs,
                                                                pos=pos_embed, query_pos=None)

            refined_masks = refined_masks.permute(1, 2, 0).view(bs_f, self.d_model, self.num_frames, m_h * m_w)
            refined_masks = refined_masks.permute(0, 2, 1, 3).reshape(bs_f, self.num_frames, self.d_model, m_h,
                                                                      m_w).flatten(0, 1)
            outputs_seg_masks = refined_masks
        else:
            outputs_seg_masks = outputs_seg_masks.squeeze(0)

        outputs_seg_masks = outputs_seg_masks.sigmoid().permute(1,0,2,3)
        outputs_seg_masks_lprop = outputs_seg_masks
        for i in range(self.stacked):
            if self.feat_loc == 'late':
                outputs_seg_masks_lprop = self.label_propagator(outputs_seg_masks_lprop, seg_feats).squeeze(1)
            elif self.feat_loc == 'attmaps_only':
                outputs_seg_masks_lprop = self.label_propagator(outputs_seg_masks_lprop, seg_feats[:, -8:]).squeeze(1)
            elif self.feat_loc == 'early_coarse':
                early_coarse_feats = features[-1].tensors
                outputs_seg_masks_lprop = self.label_propagator(outputs_seg_masks_lprop, early_coarse_feats).squeeze(1)
            elif self.feat_loc == 'early_fine':
                early_fine_feats = features[0].tensors.squeeze(0)
                outputs_seg_masks_lprop = self.label_propagator(outputs_seg_masks_lprop, early_fine_feats).squeeze(1)

        if self.lprop_mode in [1, 2]:
            outputs_seg_masks = torch.stack([outputs_seg_masks.permute(1,0,2,3), outputs_seg_masks_lprop], dim=0).mean(0)
        elif self.lprop_mode == 3:
            outputs_seg_masks = outputs_seg_masks_lprop

        out = {"pred_masks": outputs_seg_masks}

        # import ipdb;ipdb.set_trace()
        if not self.training and self.use_memory_mask_enc:
            # self.mask_mem.clear()
            # import ipdb;ipdb.set_trace()
            new_mem = torch.sigmoid(outputs_seg_masks.clone().detach())
            new_mem = torchvision.transforms.functional.gaussian_blur(new_mem, kernel_size=(7, 7))
            if clip_tag == clip_tag_flip:
                new_mem = new_mem.flip(-1)
            if clip_tag_no_flip in self.mask_mem:
                prev_mem = self.mask_mem[clip_tag_no_flip]
                # prev_shape = prev_mem.shape[2:]
                # new_shape = new_mem.shape[2:]
                save_shape = max(prev_mem.shape[-2:], new_mem.shape[-2:])
                if new_mem.shape[-2:] != save_shape:
                    new_mem = F.interpolate(new_mem, size=save_shape)
                if prev_mem.shape[-2:] != save_shape:
                    prev_mem = F.interpolate(prev_mem, size=save_shape)
                new_mem = 0.8 * new_mem + 0.2 * prev_mem
                # TODO naive approach, check and improvise this later
            self.mask_mem[clip_tag_no_flip] = new_mem
        return out



def build_swin_s_backbone(_swin_s_pretrained_path):
    print('creating swin-s-3d backbone>>>')
    logger.debug('creating swin-s-3d backbone>>>')
    swin = SwinTransformer3D(pretrained=None,
                             pretrained2d=True,
                             patch_size=(1, 4, 4),
                             in_chans=3,
                             embed_dim=96,
                             depths=(2, 2, 18, 2),
                             num_heads=(3, 6, 12, 24),
                             window_size=(8, 7, 7),
                             mlp_ratio=4.,
                             qkv_bias=True,
                             qk_scale=None,
                             drop_rate=0.,
                             attn_drop_rate=0.,
                             drop_path_rate=0.1,
                             norm_layer=nn.LayerNorm,
                             patch_norm=True,
                             frozen_stages=-1,
                             use_checkpoint=False)
    checkpoint = torch.load(_swin_s_pretrained_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    state_dict['backbone.patch_embed.proj.weight'] = state_dict['backbone.patch_embed.proj.weight'][:, :, 0:1, :, :]
    state_dict['norm_layers.3.weight'] = state_dict['backbone.norm.weight'].clone().detach()
    state_dict['norm_layers.3.bias'] = state_dict['backbone.norm.bias'].clone().detach()
    state_dict['norm_layers.2.weight'] = state_dict['backbone.norm.weight'].clone().detach()
    state_dict['norm_layers.2.bias'] = state_dict['backbone.norm.bias'].clone().detach()
    state_dict['norm_layers.1.weight'] = state_dict['backbone.norm.weight'][:384].clone().detach()
    state_dict['norm_layers.1.bias'] = state_dict['backbone.norm.bias'][:384].clone().detach()
    state_dict['norm_layers.0.weight'] = state_dict['backbone.norm.weight'][:192].clone().detach()
    state_dict['norm_layers.0.bias'] = state_dict['backbone.norm.bias'][:192].clone().detach()
    del state_dict['backbone.norm.weight']
    del state_dict['backbone.norm.bias']
    del state_dict['cls_head.fc_cls.weight']
    del state_dict['cls_head.fc_cls.bias']
    ckpt_keys = [k for k in state_dict.keys()]
    del_keys = []
    for kk in ckpt_keys:
        if 'backbone.' in kk:
            state_dict[kk.replace('backbone.', '')] = state_dict[kk]
            del_keys.append(kk)
    for kk in del_keys:
        del state_dict[kk]
    print('len(state_dict): %d' % len(state_dict))
    print('len(swin_b.state_dict()): %d' % len(swin.state_dict()))
    swin.load_state_dict(state_dict)
    return swin


def build_swin_b_backbone(_swin_b_pretrained_path):
    print('build_swin_b_backbone>>')
    logger.debug('build_swin_b_backbone>>')
    swin = SwinTransformer3D(pretrained=None,
                             pretrained2d=True,
                             patch_size=(1, 4, 4),
                             in_chans=3,
                             embed_dim=128,
                             depths=(2, 2, 18, 2),
                             num_heads=(4, 8, 16, 32),
                             window_size=(8, 7, 7),
                             mlp_ratio=4.,
                             qkv_bias=True,
                             qk_scale=None,
                             drop_rate=0.,
                             attn_drop_rate=0.,
                             drop_path_rate=0.1,
                             norm_layer=nn.LayerNorm,
                             patch_norm=True,
                             frozen_stages=-1,
                             use_checkpoint=False)
    if os.path.exists(_swin_b_pretrained_path):
        checkpoint = torch.load(_swin_b_pretrained_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        state_dict['backbone.patch_embed.proj.weight'] = state_dict['backbone.patch_embed.proj.weight'][:, :, 0:1, :, :]
        state_dict['norm_layers.3.weight'] = state_dict['backbone.norm.weight'].clone().detach()
        state_dict['norm_layers.3.bias'] = state_dict['backbone.norm.bias'].clone().detach()
        state_dict['norm_layers.2.weight'] = state_dict['backbone.norm.weight'].clone().detach()
        state_dict['norm_layers.2.bias'] = state_dict['backbone.norm.bias'].clone().detach()
        state_dict['norm_layers.1.weight'] = state_dict['backbone.norm.weight'][:512].clone().detach()
        state_dict['norm_layers.1.bias'] = state_dict['backbone.norm.bias'][:512].clone().detach()
        state_dict['norm_layers.0.weight'] = state_dict['backbone.norm.weight'][:256].clone().detach()
        state_dict['norm_layers.0.bias'] = state_dict['backbone.norm.bias'][:256].clone().detach()
        del state_dict['backbone.norm.weight']
        del state_dict['backbone.norm.bias']
        del state_dict['cls_head.fc_cls.weight']
        del state_dict['cls_head.fc_cls.bias']
        ckpt_keys = [k for k in state_dict.keys()]
        del_keys = []
        for kk in ckpt_keys:
            if 'backbone.' in kk:
                state_dict[kk.replace('backbone.', '')] = state_dict[kk]
                del_keys.append(kk)
        for kk in del_keys:
            del state_dict[kk]
        print('len(state_dict): %d' % len(state_dict))
        print('len(swin_b.state_dict()): %d' % len(swin.state_dict()))
        matched_keys = [k for k in state_dict.keys() if k in swin.state_dict().keys()]
        print('matched keys:%d' % len(matched_keys))
        swin.load_state_dict(state_dict, strict=False)
    else:
        print("Pretrained checkpoint not found at ", _swin_b_pretrained_path)
    return swin


def get_n_dec_layers(transformer, i):
    cks = [k for k in transformer.state_dict().keys() if 'decoder.%d'%i in k and 'layers' in k]
    ndec = -1
    for ck in cks:
        n = int(ck.split('layers.')[1].split('.')[0])
        if n > ndec:
            ndec = n
    return ndec + 1

def pretrain_swin(transformer, args):
    wcp_enc_upper_stages = False  # TODO check
    checkpoint = torch.load(args.resnet101_coco_weights_path, map_location='cpu')['model']
    ckpt_keys = [k for k in checkpoint.keys()]
    del_keys_1 = [k for k in checkpoint.keys() if 'vistr.backbone.' in k]
    for kk in del_keys_1:
        del checkpoint[kk]
    # print('after removing backbone keys: len(checkpoint): %d' % len(checkpoint))
    ckpt_keys = [k for k in checkpoint.keys()]
    del_keys_2 = ['vistr.query_embed.weight', 'vistr.input_proj.weight', 'vistr.input_proj.bias']
    for kk in ckpt_keys:
        if 'vistr.class' in kk:
            del_keys_2.append(kk)
        if 'vistr.bbox' in kk:
            del_keys_2.append(kk)
        if 'mask_head.' in kk:
            # checkpoint[kk.replace('mask_head.', 'fpn.')] = checkpoint[kk]
            del_keys_2.append(kk)
        if 'vistr.transformer.' in kk:
            checkpoint[kk.replace('vistr.transformer.', '')] = checkpoint[kk]
            del_keys_2.append(kk)
    for kk in del_keys_2:
        del checkpoint[kk]

    # Copy decoder layer 6 weights to initialize next decoders
    if args.dec_layers > 6 and not args.finetune and args.decoder_type == 'multiscale_query':
        cks = [k for k in checkpoint.keys() if 'decoder.layers.5.' in k]
        for i in range(6, args.dec_layers):
            for ck in cks:
                mk = ck.replace('5', str(i))
                checkpoint[mk] = checkpoint[ck].clone().detach()
    # Copy decoder layers to both multiscale query and multiscale memory decoders
    elif args.decoder_type in ['multiscale_query_memory', 'multiscale_query_memory_eff', 'multiscale_query_memory_nobidir']:
        dec_cks = [k for k in checkpoint.keys() if 'decoder.layers' in k]
        lastdec_cks = [k for k in checkpoint.keys() if 'decoder.layers.5.' in k]
        dec_norm_cks = [k for k in checkpoint.keys() if 'decoder.norm' in k]

        for i in range(2): # 2 is for multiscale query + memory
            for ck in dec_cks:
                mk = ck.replace('decoder.layers', 'decoder.%d.layers'%i)
                checkpoint[mk] = checkpoint[ck].clone().detach()

            for ck in dec_norm_cks:
                mk = ck.replace('decoder.norm', 'decoder.%d.norm'%i)
                checkpoint[mk] = checkpoint[ck].clone().detach()

            n_dec_layers = get_n_dec_layers(transformer, i)
            if i == 0:
                assert n_dec_layers == args.dec_layers, "Wrong number of decoder layers in first stage"
            for j in range(6, n_dec_layers):
                for ck in lastdec_cks:
                    mk = ck.replace('decoder.layers', 'decoder.%d.layers'%i).replace('5', str(j))
                    checkpoint[mk] = checkpoint[ck].clone().detach()

        for ck in dec_cks:
            del checkpoint[ck]
        for ck in dec_norm_cks:
            del checkpoint[ck]

    enc_del_keys = []
    enc_add_weights = {}
    for enc_stage in range(len(args.enc_layers)):
        # import ipdb; ipdb.set_trace()
        if enc_stage == 0:
            for lr_id in range(args.enc_layers[0]):
                if lr_id < 5:
                    cks = [k for k in checkpoint.keys() if 'encoder.layers.%d' % lr_id in k]
                    for ck in cks:
                        mk = ck.replace('encoder.layers.%d' % lr_id, 'encoder.layers.0.%d' % lr_id)
                        enc_add_weights[mk] = checkpoint[ck].clone().detach()
                        enc_del_keys.append(ck)
                else:
                    cks = [k for k in checkpoint.keys() if 'encoder.layers.5' in k]
                    for ck in cks:
                        mk = ck.replace('encoder.layers.5', 'encoder.layers.0.%d' % lr_id)
                        enc_add_weights[mk] = checkpoint[ck].clone().detach()
        elif wcp_enc_upper_stages:
            # import ipdb;ipdb.set_trace()
            for lr_id in range(args.enc_layers[enc_stage]):
                cks = [k for k in checkpoint.keys() if 'encoder.layers.%d' % lr_id in k]
                for ck in cks:
                    if 'norm' in ck:
                        continue
                    mk = ck.replace('encoder.layers.%d' % lr_id, 'encoder.layers.%d.%d' % (enc_stage, lr_id))
                    enc_add_weights[mk] = checkpoint[ck].clone().detach()
    if args.encoder_cross_layer and False:
        cks = [k for k in checkpoint.keys() if 'encoder.layers.0.' in k]
        for ck in cks:
            if 'norm' in ck:
                continue
            mk = ck.replace('layers.0.', 'cross_res_layers.0.')
            enc_add_weights[mk] = checkpoint[ck].clone().detach()
            # mk = ck.replace('layers.0.', 'cross_res_layers.1.')
    # import ipdb; ipdb.set_trace()
    for dk in enc_del_keys:
        del checkpoint[dk]
    for kk,vv in enc_add_weights.items():
        checkpoint[kk]=vv
    # print('len(state_dict): %d' % len(checkpoint))
    # print('len(transformer.state_dict()): %d' % len(transformer.state_dict()))
    matched_keys = [k for k in checkpoint.keys() if k in transformer.state_dict().keys()]
    # print('matched keys:%d' % len(matched_keys))
    # import ipdb;ipdb.set_trace()
    shape_mismatch = []
    # import re
    use_partial_match = True
    for kid, kk in enumerate(matched_keys):
        # if kid>66:
        # import ipdb;ipdb.set_trace()
        # print('kid:%d kk:%s'%(kid, kk))
        if checkpoint[kk].shape != transformer.state_dict()[kk].shape:
            # print('shape not matched key:%s'%kk)
            # TODO check with partial copy
            # import ipdb;ipdb.set_trace()
            if not use_partial_match:
                shape_mismatch.append(kk)
                continue
            if 'encoder.' in kk:
                if 'linear1.weight' in kk:
                    checkpoint[kk] = checkpoint[kk][:args.dim_feedforward, :].clone().detach()
                elif 'linear1.bias' in kk:
                    checkpoint[kk] = checkpoint[kk][:args.dim_feedforward].clone().detach()
                elif 'linear2.weight' in kk:
                    checkpoint[kk] = checkpoint[kk][:, :args.dim_feedforward].clone().detach()
                else:
                    import ipdb;
                    ipdb.set_trace()
                    shape_mismatch.append(kk)
            elif 'decoder.' in kk:
                if 'linear1.weight' in kk:
                    checkpoint[kk] = checkpoint[kk][:args.dim_feedforward, :].clone().detach()
                elif 'linear1.bias' in kk:
                    checkpoint[kk] = checkpoint[kk][:args.dim_feedforward].clone().detach()
                elif 'linear2.weight' in kk:
                    checkpoint[kk] = checkpoint[kk][:, :args.dim_feedforward].clone().detach()
                else:
                    import ipdb;
                    ipdb.set_trace()
                    shape_mismatch.append(kk)
            else:
                import ipdb;
                ipdb.set_trace()
                shape_mismatch.append(kk)
                # print('here')
    print('len(shape_mismatch):%d' % len(shape_mismatch))
    for kk in shape_mismatch:
        del checkpoint[kk]
    transformer.load_state_dict(checkpoint, strict=False)
    shape_matched_keys = [k for k in checkpoint.keys() if k in transformer.state_dict().keys()]
    print('shape_matched keys:%d' % len(shape_matched_keys))

    shape_matched_keys = [k for k in checkpoint.keys() if k in transformer.state_dict().keys() and 'decoder' in k]
    print('shape_matched keys in decoder:%d' % len(shape_matched_keys))
    return transformer

def pretrain_resnet(model, args):
    if args.backbone == 'resnet50':
        checkpoint = torch.load(args.resnet50_coco_weights_path, map_location='cpu')['model']
    else:
        checkpoint = torch.load(args.resnet101_coco_weights_path, map_location='cpu')['model']

    ckpt_keys = [k for k in checkpoint.keys()]
    del_keys = []
    for kk in ckpt_keys:
        if 'vistr.' in kk:
            checkpoint[kk.replace('vistr.', '')] = checkpoint[kk]
            del_keys.append(kk)
    for kk in del_keys:
        del checkpoint[kk]

    # Copy decoder layer 6 weights to initialize next decoders
    if args.dec_layers > 6 and not args.finetune and args.decoder_type == 'multiscale_query':
        cks = [k for k in checkpoint.keys() if 'decoder.layers.5.' in k]
        for i in range(6, args.dec_layers):
            for ck in cks:
                mk = ck.replace('5', str(i))
                checkpoint[mk] = checkpoint[ck].clone().detach()
    # Copy decoder layers to both multiscale query and multiscale memory decoders
    elif args.decoder_type in ['multiscale_query_memory', 'multiscale_query_memory_eff', 'multiscale_query_memory_adaptive']:
        dec_cks = [k for k in checkpoint.keys() if 'decoder.layers' in k]
        lastdec_cks = [k for k in checkpoint.keys() if 'decoder.layers.5.' in k]
        dec_norm_cks = [k for k in checkpoint.keys() if 'decoder.norm' in k]

        for i in range(2): # 2 is for multiscale query + memory
            for ck in dec_cks:
                mk = ck.replace('decoder.layers', 'decoder.%d.layers'%i)
                checkpoint[mk] = checkpoint[ck].clone().detach()

            for ck in dec_norm_cks:
                mk = ck.replace('decoder.norm', 'decoder.%d.norm'%i)
                checkpoint[mk] = checkpoint[ck].clone().detach()

            n_dec_layers = get_n_dec_layers(model, i)
            if i == 0:
                assert n_dec_layers == args.dec_layers, "Wrong number of decoder layers in first stage"
            for j in range(6, n_dec_layers):
                for ck in lastdec_cks:
                    mk = ck.replace('decoder.layers', 'decoder.%d.layers'%i).replace('5', str(j))
                    checkpoint[mk] = checkpoint[ck].clone().detach()

        for ck in dec_cks:
            del checkpoint[ck]
        for ck in dec_norm_cks:
            del checkpoint[ck]

    matched_keys = [k for k in checkpoint.keys() if k in model.state_dict().keys()]
    print('matched keys:%d' % len(matched_keys))
    shape_mismatch = []
    for kid, kk in enumerate(matched_keys):
        if checkpoint[kk].shape != model.state_dict()[kk].shape:
            if 'encoder.' in kk:
                if 'linear1.weight' in kk:
                    checkpoint[kk] = checkpoint[kk][:args.dim_feedforward, :].clone().detach()
                elif 'linear1.bias' in kk:
                    checkpoint[kk] = checkpoint[kk][:args.dim_feedforward].clone().detach()
                elif 'linear2.weight' in kk:
                    checkpoint[kk] = checkpoint[kk][:, :args.dim_feedforward].clone().detach()
                else:
                    import ipdb;
                    ipdb.set_trace()
                    shape_mismatch.append(kk)
            elif 'decoder.' in kk:
                if 'linear1.weight' in kk:
                    checkpoint[kk] = checkpoint[kk][:args.dim_feedforward, :].clone().detach()
                elif 'linear1.bias' in kk:
                    checkpoint[kk] = checkpoint[kk][:args.dim_feedforward].clone().detach()
                elif 'linear2.weight' in kk:
                    checkpoint[kk] = checkpoint[kk][:, :args.dim_feedforward].clone().detach()
                else:
                    import ipdb;
                    ipdb.set_trace()
                    shape_mismatch.append(kk)
            else:
                import ipdb;
                ipdb.set_trace()
                shape_mismatch.append(kk)

    print('len(shape_mismatch):%d' % len(shape_mismatch))
    for kk in shape_mismatch:
        del checkpoint[kk]

    model.load_state_dict(checkpoint, strict=False)
    return model

def build_model_swin_medvt(args):
    print('using backbone:%s' % args.backbone)
    backbone_dims = None
    if args.backbone == 'resnet101':
        backbone = build_backbone(args)
        backbone_dims = (256, 512, 1024, 2048)
    elif args.backbone == 'swinS':
        backbone = build_swin_s_backbone(args.swin_s_pretrained_path)
        backbone_dims = (192, 384, 768, 768)
    elif args.backbone == 'swinB':
        backbone = build_swin_b_backbone(args.swin_b_pretrained_path)
        backbone_dims = (256, 512, 1024, 1024)
    else:
        raise ValueError('backbone: %s not implemented!' % args.backbone)
    # print('args.dim_feedforward:%d' % args.dim_feedforward)
    transformer = Transformer(
        num_frames=args.num_frames,
        backbone_dims=backbone_dims,
        d_model=args.hidden_dim,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=args.pre_norm,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        num_decoder_queries=args.num_queries,
        return_intermediate_dec=True,
        encoder_cross_layer=args.encoder_cross_layer,
        decoder_type=args.decoder_type)

    if 'swin' in args.backbone and os.path.exists(args.resnet101_coco_weights_path):
        transformer = pretrain_swin(transformer, args)

    if args.lprop_mode > 0:
        temporal_strides = [1] if not hasattr(args, 'temporal_strides') else args.temporal_strides
        model = VOS_SwinMEDVTLPROP(args, backbone, backbone_dims=backbone_dims, hidden_dim=args.hidden_dim,
                                   transformer=transformer, num_frames=args.num_frames,
                                   pretrain_settings=args.pretrain_settings,
                                   lprop_mode=args.lprop_mode, temporal_strides=temporal_strides,
                                   feat_loc=args.feat_loc, stacked=args.stacked_lprop)
    else:
        model = VOS_SwinMEDVT(args, backbone, backbone_dims=backbone_dims, hidden_dim=args.hidden_dim,
                              transformer=transformer, num_frames=args.num_frames,
                              temporal_strides=args.temporal_strides)

    if args.backbone == 'resnet50' or args.backbone == 'resnet101':
        model = pretrain_resnet(model, args)

    weight_dict = {"loss_mask": args.mask_loss_coef, "loss_dice": args.dice_loss_coef}
    losses = ["masks"]
    criterion = criterions.SetCriterion(weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)
    criterion.to(torch.device(args.device))
    # import ipdb; ipdb.set_trace()
    return model, criterion
