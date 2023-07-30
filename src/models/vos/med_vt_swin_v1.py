"""
MED-VT model.
Some parts of the code is taken from VisTR (https://github.com/Epiphqny/VisTR)
which was again Modified from DETR (https://github.com/facebookresearch/detr)
And modified as needed.
"""
import torchvision.ops
import copy
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import math
import logging
from src.util.misc import NestedTensor, is_main_process
from src.models import criterions
from src.models.swin.swin_transformer_3d import SwinTransformer3D
from src.util.misc import (NestedTensor, nested_tensor_from_tensor_list)

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

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()
        print('Creating FPN-> dim:%d fpn_dims:%s context_dim:%d' % (dim, str(fpn_dims), context_dim))
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

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

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

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)
        multi_scale_features.append(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)
        multi_scale_features.append(x)

        cur_fpn = self.adapter3(fpns[2])
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
                 encoder_cross_layer=False):
        super().__init__()
        if not type(num_encoder_layers) in [list, tuple]:
            num_encoder_layers = [num_encoder_layers]
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_queries = num_decoder_queries
        self.backbone_dims = backbone_dims
        self.num_frames = num_frames
        self.input_proj = nn.Conv2d(backbone_dims[-1], d_model, kernel_size=1)
        self.input_proj_2 = nn.Conv2d(backbone_dims[-2], d_model, kernel_size=1)
        # import ipdb;ipdb.set_trace()
        if sum(num_encoder_layers) > 0:
            self.use_encoder = True
            self.num_encoder_stages = len(num_encoder_layers)
            encoder_layers = []
            for i in range(len(num_encoder_layers)):
                print('Encoder stage:%d dim_feedforward:%d'%(i, dim_feedforward))
                encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                        dropout, activation, normalize_before)
                encoder_layers.append(encoder_layer)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm, use_cross_layers=encoder_cross_layer)
        else:
            self.use_encoder = False

        fpn_dims = list(backbone_dims[:-1][::-1])
        fpn_dims[0] = 384  # temporary fix, fix it later
        # import ipdb; ipdb.set_trace()
        if len(num_encoder_layers) > 1:
            for i in range(len(num_encoder_layers) - 1):
                fpn_dims[i] = d_model
        self.fpn = MaskHeadSmallConv(dim=d_model, fpn_dims=fpn_dims, context_dim=d_model)

        if num_decoder_layers > 0:
            self.use_decoder = True
            self.query_embed = nn.Embedding(num_decoder_queries, d_model)
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)
        else:
            self.use_decoder = False
        self.bbox_attention = MHAttentionMap(d_model, d_model, bbox_nhead, dropout=0.0)
        self.d_model = d_model
        self.nhead = nhead
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, pos_list, batch_size):
        # features as TCHW
        # import ipdb; ipdb.set_trace()
        t_f = self.num_frames
        query_embed = self.query_embed.weight
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()
        assert mask is not None
        src_proj = self.input_proj(src)
        n, c, s_h, s_w = src_proj.shape
        bs_f = bs // self.num_frames
        src = src_proj.reshape(bs_f, self.num_frames, c, s_h, s_w).permute(0, 2, 1, 3, 4).flatten(-2)
        mask = mask.reshape(bs_f, self.num_frames, s_h * s_w)
        pos_embed = pos_list[-1].permute(0, 2, 1, 3, 4).flatten(-2)
        src2, mask2 = features[-2].decompose()
        assert mask2 is not None
        src_proj_2 = self.input_proj_2(src2)
        n2, c2, s_h2, s_w2 = src_proj_2.shape
        src2 = src_proj_2.reshape(bs_f, self.num_frames, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
        mask2 = mask2.reshape(bs_f, self.num_frames, s_h2 * s_w2)
        pos2 = pos_list[-2].permute(0, 2, 1, 3, 4).flatten(-2)
        # import ipdb;ipdb.set_trace()
        ################################################################
        ###################################################################

        ##############################
        #################################
        bs, c, t, hw = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        bs2, c2, t2, hw2 = src2.shape
        src2 = src2.flatten(2).permute(2, 0, 1)
        pos2 = pos2.flatten(2).permute(2, 0, 1)
        mask2 = mask2.flatten(1)
        for i in range(3):
            _, c_f, h, w = features[i].tensors.shape
            features[i].tensors = features[i].tensors.reshape(bs, t, c_f, h, w)
        memory, memory2 = self.encoder(s_h, s_w, s_h2, s_w2, t_f, src, src_key_padding_mask=mask, pos=pos_embed,
                                       src2=src2, mask2=mask2, pos2=pos2)
        memory = memory.permute(1, 2, 0).view(bs, c, t, hw)
        memory = memory.permute(0, 2, 1, 3).reshape(bs, t, c, s_h, s_w).flatten(0, 1)

        memory2 = memory2.permute(1, 2, 0).view(bs2, c2, t2, hw2)
        memory2 = memory2.permute(0, 2, 1, 3).reshape(bs2, t2, c2, s_h2, s_w2).flatten(0, 1)
        fpn_features = [memory2, features[1].tensors.flatten(0, 1), features[0].tensors.flatten(0, 1)]
        ms_feats = self.fpn(memory, fpn_features)
        hr_feat = ms_feats[-1]
        dec_features = []
        pos_embed_list = []
        size_list = []
        dec_mask_list = []
        for i in range(3):
            fi = ms_feats[i]
            ni, ci, hi, wi = fi.shape
            fi = fi.reshape(bs, t, ci, hi, wi).permute(0, 2, 1, 3, 4).flatten(-2).flatten(2).permute(2, 0, 1)
            dec_mask_i = features[-1 - i].mask.reshape(bs, t, hi * wi).flatten(1)
            pe = pos_list[-1 - i].permute(0, 2, 1, 3, 4).flatten(-2).flatten(2).permute(2, 0, 1)
            dec_features.append(fi)
            pos_embed_list.append(pe)
            size_list.append((hi, wi))
            dec_mask_list.append(dec_mask_i)
        query_embed = query_embed.unsqueeze(1)
        tq, bq, cq = query_embed.shape
        query_embed = query_embed.repeat(t_f // tq, bs, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, dec_features, memory_key_padding_mask=dec_mask_list,
                          pos=pos_embed_list, query_pos=query_embed, size_list=size_list)
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
        return seg_feats


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layers, num_layers, norm=None, cross_pos=0, use_cross_layers=False):
        super().__init__()
        self.num_layers = num_layers
        self.num_stages = len(encoder_layers)
        import ipdb; ipdb.set_trace()
        # self.layers = nn.ModuleList()
        layer_list = [copy.deepcopy(encoder_layers[0]) for _ in range(num_layers[0])]
        # self.layers.append(nn.ModuleList(layer_list))
        # additional layers for additional stages
        for j in range(1,len(encoder_layers)):
            layer_list = layer_list + [copy.deepcopy(encoder_layers[j]) for _ in range(num_layers[j])]
        self.layers= nn.ModuleList(layer_list)
        self.norm = norm
        self.cross_pos = cross_pos
        self.norm2 = None if norm is None else copy.deepcopy(norm)
        self.use_cross_layers = use_cross_layers
        if use_cross_layers:
            self.cross_res_layers = nn.ModuleList([CrossResolutionEncoderLayer(384, 8, 1024)])

    def forward(self, s_h, s_w, s_h2, s_w2, t_f, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                src2=None, mask2=None, pos2=None):
        # import ipdb;ipdb.set_trace()
        output = src
        output2 = src2
        if self.use_cross_layers and (self.cross_pos == -1 or self.cross_pos == 2):
            outputx = output.permute(1, 2, 0).view(1, 384, t_f, s_h * s_w)
            outputx = outputx.permute(0, 2, 1, 3).reshape(1, t_f, 384, s_h, s_w).flatten(0, 1)
            outputx = F.interpolate(outputx, (s_h2, s_w2), mode='bilinear')
            n2, c2, s_h2, s_w2 = outputx.shape
            outputx = outputx.reshape(1, t_f, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
            outputx = outputx.flatten(2).permute(2, 0, 1)
            output2 = self.cross_res_layers[0](outputx, output2, src_key_padding_mask, mask2, pos, pos2)
        for i in range(self.num_layers[0]):
            output = self.layers[i](output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.use_cross_layers and (self.cross_pos == 0 or self.cross_pos == 2):
            outputx = output.permute(1, 2, 0).view(1, 384, t_f, s_h * s_w)
            outputx = outputx.permute(0, 2, 1, 3).reshape(1, t_f, 384, s_h, s_w).flatten(0, 1)
            outputx = F.interpolate(outputx, (s_h2, s_w2), mode='bilinear')
            n2, c2, s_h2, s_w2 = outputx.shape
            outputx = outputx.reshape(1, t_f, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
            outputx = outputx.flatten(2).permute(2, 0, 1)
            output2 = self.cross_res_layers[0](outputx, output2, src_key_padding_mask, mask2, pos, pos2)
        if self.num_stages > 1:
            # import ipdb; ipdb.set_trace()
            for i in range(self.num_layers[1]):
                output2 = self.layers[i](output2, src_mask=None, src_key_padding_mask=mask2, pos=pos2)
        if self.use_cross_layers and (self.cross_pos == 1 or self.cross_pos == 2):
            outputx = output.permute(1, 2, 0).view(1, 384, t_f, s_h * s_w)
            outputx = outputx.permute(0, 2, 1, 3).reshape(1, t_f, 384, s_h, s_w).flatten(0, 1)
            outputx = F.interpolate(outputx, (s_h2, s_w2), mode='bilinear')
            n2, c2, s_h2, s_w2 = outputx.shape
            outputx = outputx.reshape(1, t_f, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
            outputx = outputx.flatten(2).permute(2, 0, 1)
            output2 = self.cross_res_layers[0](outputx, output2, src_key_padding_mask, mask2, pos, pos2)
        if self.norm is not None:
            output = self.norm(output)
            if self.num_stages > 1:
                output2 = self.norm2(output2)
        return output, output2


class TransformerDecoder(nn.Module):

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

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu", custom_instance_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.custom_instance_norm = custom_instance_norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, src1, src2, mask1, mask2, pos1, pos2):
        kk = src1
        if self.custom_instance_norm:
            ku = src1.mean(dim=0)
            ks = src1.std(dim=0)
            qu = src2.mean(dim=0)
            qs = src2.std(dim=0)
            kk = (kk - ku)
            if ks.min() > 1e-10:
                kk = kk / ks
            kk = (kk * qs) + qu
        kp = kk + pos2
        qp = src2 + pos2
        attn_mask = torch.mm(mask2.transpose(1, 0).double(), mask2.double()).bool()
        out = self.self_attn(qp, kp, value=kk, attn_mask=attn_mask, key_padding_mask=mask2)[0]
        out = src2 + self.dropout1(out)
        out = self.norm1(out)
        out2 = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = out + self.dropout2(out2)
        out = self.norm2(out)
        return out


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
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
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
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
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


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
    def __init__(self, args, backbone, backbone_dims, hidden_dim, transformer, num_frames):
        super().__init__()
        self.backbone_name = args.backbone
        self.backbone = backbone
        self.num_frames = num_frames
        self.position_embedding = build_position_encoding(args)
        if transformer is None:
            self.input_proj = nn.Conv2d(backbone_dims[-1], hidden_dim, kernel_size=1)
        else:
            self.transformer = transformer
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
        if self.transformer is None:
            src, mask = features[-1].decompose()
            src_proj = self.input_proj(src)
            seg_feats = src_proj.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            seg_feats = self.transformer(features, pos_list, batch_size)
        mask_ins = seg_feats.unsqueeze(0).permute(0, 2, 1, 3, 4)
        outputs_seg_masks = self.insmask_head(mask_ins)
        outputs_seg_masks = outputs_seg_masks.squeeze(0)
        out = {"pred_masks": outputs_seg_masks}
        return out


class VOS_Vanilla_FPN(nn.Module):
    def __init__(self, args, backbone, backbone_dims, hidden_dim):
        super().__init__()
        self.backbone_name = args.backbone
        self.backbone = backbone
        if args.backbone == 'resnet101' or args.backbone == 'resnet50':
            self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        else:
            self.input_proj = nn.Conv3d(backbone_dims[-1], hidden_dim, kernel_size=1)
        self.use_fpn = args.use_fpn
        if args.use_fpn:
            self.fpn = MaskHeadSmallConv(dim=hidden_dim, fpn_dims=backbone_dims[:-1][::-1], context_dim=hidden_dim)
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
            features[-1] = self.input_proj(src)
            if self.use_fpn:
                ms_feats = self.fpn(features[-1], features[:-1][::-1])
                for i in range(len(features)):
                    features[i] = ms_feats[i]
                src_proj = features[0]
            else:
                src_proj = features[-1]
            src_proj = src_proj.permute(1, 0, 2, 3).unsqueeze(0)  # tchw->bcthw
        else:
            features = self.backbone(samples.tensors.permute(1, 0, 2, 3).unsqueeze(0))
            features[-1] = self.input_proj(features[-1])
            if self.use_fpn:
                for i in range(len(features)):
                    features[i] = features[i].permute(0, 2, 1, 3, 4).flatten(0, 1)
                ms_feats = self.fpn(features[-1], features[:-1][::-1])
                for i in range(len(features)):
                    features[i] = ms_feats[len(features) - i - 1]
                src_proj = features[0]
            else:
                src_proj = features[-1]
            src_proj = src_proj.permute(1, 0, 2, 3).unsqueeze(0)
        mask_ins = src_proj
        outputs_seg_masks = self.insmask_head(mask_ins)
        outputs_seg_masks = outputs_seg_masks.squeeze(0)
        out = {"pred_masks": outputs_seg_masks}
        return out


class VOS_Vanilla(nn.Module):
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
            src_proj = self.input_proj(features[-1])
        mask_ins = src_proj
        outputs_seg_masks = self.insmask_head(mask_ins)
        outputs_seg_masks = outputs_seg_masks.squeeze(0)
        out = {"pred_masks": outputs_seg_masks}
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
    swin.load_state_dict(state_dict)
    return swin


def build_model_vanilla(args):
    print('using backbone:%s' % args.backbone)
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
    model = VOS_Vanilla(args, backbone, backbone_dims=backbone_dims, hidden_dim=args.hidden_dim)
    if args.backbone == 'resnet50' or args.backbone == 'resnet101':
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
        print('len(state_dict): %d' % len(checkpoint))
        print('len(model.state_dict()): %d' % len(model.state_dict()))
        matched_keys = [k for k in checkpoint.keys() if k in model.state_dict().keys()]
        print('matched keys:%d' % len(matched_keys))
        model.load_state_dict(checkpoint, strict=False)
    weight_dict = {"loss_mask": args.mask_loss_coef, "loss_dice": args.dice_loss_coef}
    losses = ["masks"]
    criterion = criterions.SetCriterion(weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)
    criterion.to(torch.device(args.device))
    return model, criterion


def build_model_vanilla_fpn(args):
    print('using backbone:%s' % args.backbone)
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
    model = VOS_Vanilla_FPN(args, backbone, backbone_dims=backbone_dims, hidden_dim=args.hidden_dim)
    if args.backbone == 'resnet50' or args.backbone == 'resnet101':
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
        print('len(state_dict): %d' % len(checkpoint))
        print('len(model.state_dict()): %d' % len(model.state_dict()))
        matched_keys = [k for k in checkpoint.keys() if k in model.state_dict().keys()]
        print('matched keys:%d' % len(matched_keys))
        model.load_state_dict(checkpoint, strict=False)
    weight_dict = {"loss_mask": args.mask_loss_coef, "loss_dice": args.dice_loss_coef}
    losses = ["masks"]
    criterion = criterions.SetCriterion(weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)
    criterion.to(torch.device(args.device))
    return model, criterion


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
    print('args.dim_feedforward:%d'%args.dim_feedforward)
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
        encoder_cross_layer=args.encoder_cross_layer)

    if 'swin' in args.backbone:
        wcp_enc_upper_stages = False  # TODO check
        checkpoint = torch.load(args.resnet101_coco_weights_path, map_location='cpu')['model']
        print('len(checkpoint): %d' % len(checkpoint))
        ckpt_keys = [k for k in checkpoint.keys()]
        del_keys_1 = [k for k in checkpoint.keys() if 'vistr.backbone.' in k]
        for kk in del_keys_1:
            del checkpoint[kk]
        print('after removing backbone keys: len(checkpoint): %d' % len(checkpoint))
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
        if args.dec_layers > 6 and not args.finetune:
            cks = [k for k in checkpoint.keys() if 'decoder.layers.5.' in k]
            for i in range(6, args.dec_layers):
                for ck in cks:
                    mk = ck.replace('5', str(i))
                    checkpoint[mk] = checkpoint[ck].clone().detach()
        # import ipdb; ipdb.set_trace()
        enc_del_keys = []
        for enc_stage in range(len(args.enc_layers)):
            # import ipdb; ipdb.set_trace()
            if enc_stage == 0:
                for lr_id in range(args.enc_layers[0]):
                    if lr_id <5:
                        cks = [k for k in checkpoint.keys() if 'encoder.layers.%d'%lr_id in k]
                        for ck in cks:
                            mk = ck.replace('encoder.layers.%d'%lr_id, 'encoder.layers.0.%d'%lr_id) 
                            checkpoint[mk] = checkpoint[ck].clone().detach()
                            enc_del_keys.append(ck)
                    else:
                        cks = [k for k in checkpoint.keys() if 'encoder.layers.5' in k]
                        for ck in cks:
                            mk = ck.replace('encoder.layers.5', 'encoder.layers.0.%d'%lr_id) 
                            checkpoint[mk] = checkpoint[ck].clone().detach()
            elif wcp_enc_upper_stages:
                for lr_id in range(args.enc_layers[enc_stage]):
                    cks = [k for k in checkpoint.keys() if 'encoder.layers.%d'%lr_id in k]
                    for ck in cks:
                        mk = ck.replace('encoder.layers.%d'%lr_id, 'encoder.layers.%d.%d'%(enc_stage,lr_id)) 
                        checkpoint[mk] = checkpoint[ck].clone().detach()

        '''
        if sum(args.enc_layers) > 6:
            additional_lyrs = int(args.enc_layers) - 6
            for i in range(additional_lyrs):
                cks = [k for k in checkpoint.keys() if 'encoder.layers.%d.' % i in k]
                for ck in cks:
                    if 'norm' in ck or 'linear' in ck:
                        continue
                    # mk = ck.replace('%d' % (5-i), '%d' % (5+additional_lyrs-i))
                    mk = ck.replace('%d' % i, '%d' % (i + 6))
                    checkpoint[mk] = checkpoint[ck].clone().detach()
        '''
        if args.encoder_cross_layer:
            cks = [k for k in checkpoint.keys() if 'encoder.layers.0.' in k]
            for ck in cks:
                if 'norm' in ck:
                    continue
                mk = ck.replace('layers.0.', 'cross_res_layers.0.')
                checkpoint[mk] = checkpoint[ck].clone().detach()
                # mk = ck.replace('layers.0.', 'cross_res_layers.1.')
        # import ipdb; ipdb.set_trace()
        for dk in enc_del_keys:
            del checkpoint[dk]
        print('len(state_dict): %d' % len(checkpoint))
        print('len(transformer.state_dict()): %d' % len(transformer.state_dict()))
        matched_keys = [k for k in checkpoint.keys() if k in transformer.state_dict().keys()]
        print('matched keys:%d' % len(matched_keys))
        # import ipdb;ipdb.set_trace()
        shape_mismatch = []
        # import re
        use_partial_match= True
        for kid, kk in enumerate(matched_keys):
            # if kid>66:
                # import ipdb;ipdb.set_trace()
                # print('kid:%d kk:%s'%(kid, kk))
            if checkpoint[kk].shape != transformer.state_dict()[kk].shape:
                # print('shape not matched key:%s'%kk)
                #TODO check with partial copy
                # import ipdb;ipdb.set_trace()
                if not use_partial_match:
                    shape_mismatch.append(kk)
                    continue

                if 'encoder.' in kk:
                    if  'linear1.weight' in kk:
                        checkpoint[kk] = checkpoint[kk][:args.dim_feedforward,:].clone().detach()
                    elif 'linear1.bias' in kk:
                        checkpoint[kk] = checkpoint[kk][:args.dim_feedforward].clone().detach()
                    elif 'linear2.weight' in kk:
                        checkpoint[kk] = checkpoint[kk][:, :args.dim_feedforward].clone().detach()
                    else:
                        import ipdb; ipdb.set_trace()
                        shape_mismatch.append(kk)
                elif 'decoder.' in kk:
                    if  'linear1.weight' in kk:
                        checkpoint[kk] = checkpoint[kk][:args.dim_feedforward,:].clone().detach()
                    elif 'linear1.bias' in kk:
                        checkpoint[kk] = checkpoint[kk][:args.dim_feedforward].clone().detach()
                    elif 'linear2.weight' in kk:
                        checkpoint[kk] = checkpoint[kk][:, :args.dim_feedforward].clone().detach()
                    else:
                        import ipdb; ipdb.set_trace()
                        shape_mismatch.append(kk)
                else:
                    import ipdb;ipdb.set_trace()
                    shape_mismatch.append(kk)
                    # print('here')
        print('len(shape_mismatch):%d'%len(shape_mismatch))
        for kk in shape_mismatch:
            del checkpoint[kk]
        transformer.load_state_dict(checkpoint, strict=False)
        shape_matched_keys = [k for k in checkpoint.keys() if k in transformer.state_dict().keys()]
        print('shape_matched keys:%d' % len(shape_matched_keys))
 
    model = VOS_SwinMEDVT(args, backbone, backbone_dims=backbone_dims, hidden_dim=args.hidden_dim,
                          transformer=transformer, num_frames=args.num_frames)
    if args.backbone == 'resnet50' or args.backbone == 'resnet101':
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
        print('len(state_dict): %d' % len(checkpoint))
        print('len(model.state_dict()): %d' % len(model.state_dict()))
        matched_keys = [k for k in checkpoint.keys() if k in model.state_dict().keys()]
        print('matched keys:%d' % len(matched_keys))
        model.load_state_dict(checkpoint, strict=False)

    weight_dict = {"loss_mask": args.mask_loss_coef, "loss_dice": args.dice_loss_coef}
    losses = ["masks"]
    criterion = criterions.SetCriterion(weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)
    criterion.to(torch.device(args.device))
    print('build model done>>>>')
    return model, criterion


def build_model(args):
    """
    backbone = build_backbone(args)
    transformer = Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        num_encoder_stages=2,
        num_decoder_stages=3,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )
    model = VisTR(
        backbone,
        transformer,
        num_classes=1,
        num_frames=args.num_frames,
        num_queries=args.num_queries,
    )
    model = MEDVT(model)
    """
    swin_b = build_swin_b_backbone(args.swin_b_pretrained_path)
    model = VOS_MEDVT(swin_b)
    # Load pretrained transformer weights from detr model
    # TODO
    import ipdb;
    ipdb.set_trace()
    checkpoint = torch.load(args.pretrained_weights, map_location='cpu')['model']
    model_keys = model.state_dict().keys()
    # import ipdb;
    # ipdb.set_trace()
    if args.dec_layers > 6 and not args.finetune:
        cks = [k for k in checkpoint.keys() if 'vistr.transformer.decoder.layers.5.' in k]
        for i in range(6, args.dec_layers):
            for ck in cks:
                mk = ck.replace('5', str(i))
                checkpoint[mk] = checkpoint[ck].clone().detach()
    # import ipdb;
    # ipdb.set_trace()
    # if 'vistr.transformer.encoder.cross_res_layers.0.':
    cks = [k for k in checkpoint.keys() if 'vistr.transformer.encoder.layers.0.' in k]
    for ck in cks:
        # if 'norm' in ck:
        #    continue
        mk = ck.replace('layers.0.', 'cross_res_layers.0.')
        checkpoint[mk] = checkpoint[ck].clone().detach()
        # mk = ck.replace('layers.0.', 'cross_res_layers.1.')
        # checkpoint[mk] = checkpoint[ck].clone().detach()

    if int(args.enc_layers) > 6:
        additional_lyrs = int(args.enc_layers) - 6
        for i in range(additional_lyrs):
            cks = [k for k in checkpoint.keys() if 'vistr.transformer.encoder.layers.%d.' % i in k]
            for ck in cks:
                # mk = ck.replace('%d' % (5-i), '%d' % (5+additional_lyrs-i))
                mk = ck.replace('%d' % i, '%d' % (i + 6))
                checkpoint[mk] = checkpoint[ck].clone().detach()
    del_keys = []
    for k in checkpoint.keys():
        if k not in model_keys:
            del_keys.append(k)
            continue
        if checkpoint[k].shape != model.state_dict()[k].shape:
            del_keys.append(k)
            continue
    for k in del_keys:
        del checkpoint[k]
    import ipdb;
    ipdb.set_trace()
    model.load_state_dict(checkpoint, strict=False)
    # </ Model weights loading done>
    weight_dict = {"loss_mask": args.mask_loss_coef, "loss_dice": args.dice_loss_coef}
    losses = ["masks"]
    criterion = criterions.SetCriterion(weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)
    criterion.to(torch.device(args.device))
    return model, criterion
