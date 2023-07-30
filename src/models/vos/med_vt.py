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
from src.models import criterions
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
        return x, multi_scale_features


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

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, num_encoder_stages=2, num_decoder_stages=3,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        encoder_layers = []
        for i in range(num_encoder_stages):
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward // (2 ** i),
                                                    dropout, activation, normalize_before)
            encoder_layers.append(encoder_layer)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self.mask_head = MaskHeadSmallConv(384, [384, 512, 256], 384)
        self.d_model = d_model
        self.nhead = nhead
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, s_h, s_w, s_h2, s_w2, t_f, src, mask, query_embed, pos_embed, src2, mask2, pos2, features,
                pos_list):
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
        hr_feat, ms_feats = self.mask_head(memory,
                                           [memory2, features[1].tensors.flatten(0, 1),
                                            features[0].tensors.flatten(0, 1)])
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
        return hs.transpose(1, 2), hr_feat


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layers, num_layers, norm=None, cross_pos=0):
        super().__init__()
        self.num_layers = num_layers
        self.num_stages = len(encoder_layers)
        layer_list = [copy.deepcopy(encoder_layers[0]) for _ in range(6)]
        layer_list = layer_list + [copy.deepcopy(encoder_layers[1]) for _ in range(1)]
        self.layers = nn.ModuleList(layer_list)
        self.norm = norm
        self.cross_pos = cross_pos
        self.norm2 = None if norm is None else copy.deepcopy(norm)
        self.cross_res_layers = nn.ModuleList([CrossResolutionEncoderLayer(384, 8, 1024)])

    def forward(self, s_h, s_w, s_h2, s_w2, t_f, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                src2=None, mask2=None, pos2=None):
        output = src
        output2 = src2
        if self.cross_pos ==-1 or self.cross_pos ==2:
            outputx = output.permute(1, 2, 0).view(1, 384, t_f, s_h * s_w)
            outputx = outputx.permute(0, 2, 1, 3).reshape(1, t_f, 384, s_h, s_w).flatten(0, 1)
            outputx = F.interpolate(outputx, (s_h2, s_w2), mode='bilinear')
            n2, c2, s_h2, s_w2 = outputx.shape
            outputx = outputx.reshape(1, t_f, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
            outputx = outputx.flatten(2).permute(2, 0, 1)
            output2 = self.cross_res_layers[0](outputx, output2, src_key_padding_mask, mask2, pos, pos2)
        for i in range(6):
            output = self.layers[i](output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.cross_pos ==0 or self.cross_pos ==2:
            outputx = output.permute(1, 2, 0).view(1, 384, t_f, s_h * s_w)
            outputx = outputx.permute(0, 2, 1, 3).reshape(1, t_f, 384, s_h, s_w).flatten(0, 1)
            outputx = F.interpolate(outputx, (s_h2, s_w2), mode='bilinear')
            n2, c2, s_h2, s_w2 = outputx.shape
            outputx = outputx.reshape(1, t_f, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
            outputx = outputx.flatten(2).permute(2, 0, 1)
            output2 = self.cross_res_layers[0](outputx, output2, src_key_padding_mask, mask2, pos, pos2)
        for i in range(6, 7):
            output2 = self.layers[i](output2, src_mask=None, src_key_padding_mask=mask2, pos=pos2)
        if self.cross_pos == 1 or self.cross_pos == 2:
            outputx = output.permute(1, 2, 0).view(1, 384, t_f, s_h * s_w)
            outputx = outputx.permute(0, 2, 1, 3).reshape(1, t_f, 384, s_h, s_w).flatten(0, 1)
            outputx = F.interpolate(outputx, (s_h2, s_w2), mode='bilinear')
            n2, c2, s_h2, s_w2 = outputx.shape
            outputx = outputx.reshape(1, t_f, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
            outputx = outputx.flatten(2).permute(2, 0, 1)
            output2 = self.cross_res_layers[0](outputx, output2, src_key_padding_mask, mask2, pos, pos2)
        if self.norm is not None:
            output = self.norm(output)
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

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu", hard_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.hard_norm = hard_norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, src1, src2, mask1, mask2, pos1, pos2):
        kk = src1
        if self.hard_norm:
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


def build_model(args):
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
    weight_dict = {"loss_mask": args.mask_loss_coef, "loss_dice": args.dice_loss_coef}
    losses = ["masks"]
    criterion = criterions.SetCriterion(weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)
    criterion.to(torch.device(args.device))
    return model, criterion
