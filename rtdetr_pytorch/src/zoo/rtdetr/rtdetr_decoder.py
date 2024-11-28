"""by lyuwenyu
"""

import math 
import copy 
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob


from src.core import register


__all__ = ['RTDETRTransformer']



class MLP(nn.Module):
    # 256, 256, 4, 3
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        # [256] * (3 - 1) --> [256, 256]
        h = [hidden_dim] * (num_layers - 1)
        # [256] + [256, 256] --> [256, 256, 256]
        # [256, 256] + [4] --> [256, 256, 4]
        # (256, 256) (256, 256) (256, 4) 三层 Linear
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        # [0 1 2] < 2 --> 0 1 会经过 act，而 2 即最后一层不经过 act 激活函数
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4,):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()


    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        # 2, 500 <-- (2, 500, 256)
        bs, Len_q = query.shape[:2]
        # 6804 <-- (2, 6804, 256)
        Len_v = value.shape[1]

        # (2, 6804, 256) Linear --> (2, 6804, 256)
        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        # (2, 6804, 256) --> (2, 6804, 8, 32)
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        # (2, 500, 256) Linear --> (2, 500, 192(8*3*4*2)) --> (2, 500, 8, 3, 4, 2)
        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        # (2, 500, 256) Linear --> (2, 500, 96(8*3*4)) --> (2, 500, 8, 12(3*4))
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        # (2, 500, 8, 12) --> (2, 500, 8, 12) --> (2, 500, 8, 3, 4)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        # 4 <-- (2, 500, 1, 4)
        # xy 通常表示单一位置
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(
                bs, Len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer
        # xywh 位置和尺度，通常表示一个框区域
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                # (2, 500, 1, 4) --> (2, 500, 1, 1, 1, 2) xy
                # (2, 500, 1, 4) --> (2, 500, 1, 1, 1, 2) wh
                # (2, 500, 8, 3, 4, 2) <-- (2, 500, 1, 1, 1, 2) xy + (2, 500, 8, 3, 4, 2) / 4 * (2, 500, 1, 1, 1, 2) wh * 0.5
                # x = x_reference + (x_offset / 4) * (w / 2)
                # y = y_reference + (y_offset / 4) * (h / 2)
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        # (2, 500, 256) <-- (2, 6804, 8, 32), [[80, 80], [40, 40], [20, 20]], (2, 500, 8, 3, 4, 2), (2, 500, 8, 3, 4)
        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)
        # (2, 500, 256) Linear --> (2, 500, 256)
        output = self.output_proj(output)
        # (2, 500, 256)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,):
        # 初始化参数 256, 8, 1024, 0., 'relu', 3, 4
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        # 256, 8, 3, 4
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # self._reset_parameters()

    # def _reset_parameters(self):
    #     linear_init_(self.linear1)
    #     linear_init_(self.linear2)
    #     xavier_uniform_(self.linear1.weight)
    #     xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,  # init_ref_points_unact --> reference_points
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        # (2, 500, 256) + (2, 500, 256) --> (2, 500, 256)
        q = k = self.with_pos_embed(tgt, query_pos_embed)

        # if attn_mask is not None:
        #     attn_mask = torch.where(
        #         attn_mask.to(torch.bool),
        #         torch.zeros_like(attn_mask),
        #         torch.full_like(attn_mask, float('-inf'), dtype=tgt.dtype))

        # (2, 500, 256)
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        # (2, 500, 256) + (2, 500, 256) --> (2, 500, 256)
        tgt = tgt + self.dropout1(tgt2)
        # (2, 500, 256) --> (2, 500, 256)
        tgt = self.norm1(tgt)

        # cross attention
        # (2, 500, 256) --> (2, 500, 256)
        tgt2 = self.cross_attn(\
            self.with_pos_embed(tgt, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        # (2, 500, 256) + (2, 500, 256) --> (2, 500, 256)
        tgt = tgt + self.dropout2(tgt2)
        # (2, 500, 256) --> (2, 500, 256)
        tgt = self.norm2(tgt)

        # ffn
        # (2, 500, 256) --> (2, 500, 256)
        tgt2 = self.forward_ffn(tgt)
        # (2, 500, 256) + (2, 500, 256) --> (2, 500, 256)
        tgt = tgt + self.dropout4(tgt2)
        # (2, 500, 256) --> (2, 500, 256)
        tgt = self.norm3(tgt)

        # (2, 500, 256）
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None):  # 由于输入图像的尺寸是一样的，所以编码后的形状也是一样的，故没有 memory_mask
        # (2, 500, 256)
        output = tgt
        # 保存每个 decoder 层预测的 bbox 和 cls
        dec_out_bboxes = []
        dec_out_logits = []
        # 为了不影响 IOU-aware query selection 部分的参数，Anchor Boxes，即 ref_points_unact 已经 detach()
        # (2, 500, 4) --> (2, 500, 4)
        ref_points_detach = F.sigmoid(ref_points_unact)

        # 6 层 decoder
        for i, layer in enumerate(self.layers):
            # (2, 500, 4) --> (2, 500, 1, 4)
            ref_points_input = ref_points_detach.unsqueeze(2)
            # (2, 500, 4) --> (2, 500, 256)
            query_pos_embed = query_pos_head(ref_points_detach)
            # (2, 500, 256)
            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            # (2, 500, 256) MLP --> (2, 500, 4)
            # (2, 500, 4) --> (2, 500, 4)
            # (2, 500, 4) + (2, 500, 4) --> (2, 500, 4) --> (2, 500, 4)
            # 中间层 bbox 预测输出
            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            # 训练模式
            if self.training:
                # (2, 500, 256) Linear --> (2, 500, 80)
                # [(2, 500, 80)]
                # [(2, 500, 80), (2, 500, 80)]
                # ... ...
                # [(2, 500, 80), (2, 500, 80), (2, 500, 80), (2, 500, 80), (2, 500, 80), (2, 500, 80)]
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    # [(2, 500, 4)]
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    # 1, 2, 3, 4, 5
                    # 使用经过上一层调整后的 ref_points_detach --> inter_ref_bbox ==> ref_points
                    # (2, 500, 256) MLP --> (2, 500, 4)
                    # (2, 500, 4) --> (2, 500, 4)
                    # (2, 500, 4) + (2, 500, 4) --> (2, 500, 4) --> (2, 500, 4)
                    #
                    # [(2, 500, 4)] * 6
                    # [(2, 500, 4), (2, 500, 4), (2, 500, 4), (2, 500, 4), (2, 500, 4), (2, 500, 4)]
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
            # 验证模式
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break
            # 当前层的 inter_ref_bbox 作为下一层的 ref_points
            # (2, 500, 4)
            ref_points = inter_ref_bbox
            # 中间层输出值不进行参数更新，可以作为结果直接使用
            # (2, 500, 4) detach --> (2, 500, 4)
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox
        # (6, 2, 500, 4) <-- [(2, 500, 4), (2, 500, 4), (2, 500, 4), (2, 500, 4), (2, 500, 4), (2, 500, 4)]
        # (6, 2, 500, 80) <-- [(2, 500, 80), (2, 500, 80), (2, 500, 80), (2, 500, 80), (2, 500, 80), (2, 500, 80)]
        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


@register
class RTDETRTransformer(nn.Module):
    __share__ = ['num_classes']
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True):

        super(RTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        # 256, 8, 1024, 0., 'relu', 3, 4
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        # 256, TransFormerDecoderLayer, 6, -1
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        # denoising part
        if num_denoising > 0: 
            # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim,)
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        
        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)


    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        # [256, 256, 256]
        # input_proj 保存三个相同的 conv(256, 256, 1) + norm(256)
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )
        # 如果 num_levels 比 len(feat_channels) 则需要填充 input_proj 以满足 num_levels 个封装的conv1x1
        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        # [(2, 256, 72, 72), (2, 256, 36, 36), (2, 256, 18, 18)] 经 input_roj 映射后尺度不变
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        # [(2, 256, 72, 72), (2, 256, 36, 36), (2, 256, 18, 18)]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            # (2, 256, 72, 72) --> (2, 256, 5184) --> (2, 5184, 256)
            # (2, 256, 36, 36) --> (2, 256, 1296) --> (2, 1296, 256)
            # (2, 256, 18, 18) --> (2, 256, 324) --> (2, 324, 256)
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            # [(72, 72)]
            # [(72, 72), (36, 36)]
            # [(72, 72), (36, 36), (18, 18)]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            # [0, 5184]
            # [0, 5184, 6480]
            # [0, 5184, 6480, 6804]
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        # [2, 6804, 256]
        feat_flatten = torch.concat(feat_flatten, 1)
        # [0, 5184, 6480, 6804] --> [0, 5184, 6480]
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        # [(72, 72), (36, 36), (18, 18)]
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        # [(72, 72), (36, 36), (18, 18)]
        for lvl, (h, w) in enumerate(spatial_shapes):
            # (72, 72), (72, 72) 行与列，前者每个元素都是行的索引，后者每个元素都是列的索引
            # 0 0 0 ...  0 1 2 ...  ==>  [0, 0] [0, 1] [0, 2] ...
            # 1 1 1 ...  0 1 2 ...       [1, 0] [1, 1] [1, 2] ...
            # 2 2 2 ...  0 1 2 ...       [2, 0] [2, 1] [2, 2] ...
            grid_y, grid_x = torch.meshgrid(\
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')
            # (72, 72, 2)
            grid_xy = torch.stack([grid_x, grid_y], -1)
            # (2,): [72, 72]
            valid_WH = torch.tensor([w, h]).to(dtype)
            # (72, 72, 2) --> (1, 72, 72, 2) 将坐标轴从元素的边缘偏移至中心，并将每个元素值都标准化
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            # (1, 72, 72, 2) 元素值 0.05 * 1 = 0.05
            # (1, 36, 36, 2) 元素值 0.05 * 2 = 0.1
            # (1, 18, 18, 2) 元素值 0.05 * 4 = 0.2
            # 在每个特征层上 anchor 的宽高
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            # (1, 72, 72, 2) concat (1, 72, 72, 2) --> (1, 72, 72, 4) --> (1, 5184, 4)
            # (1, 36, 36, 2) concat (1, 36, 36, 2) --> (1, 36, 36, 4) --> (1, 1296, 4)
            # (1, 18, 18, 2) concat (1, 18, 18, 2) --> (1, 18, 18, 4) --> (1, 324, 4)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))
        # (1, 6804, 4) 所有特征层的 anchor 信息（坐标 + 宽高）
        anchors = torch.concat(anchors, 1).to(device)
        # (1, 6804, 4) * (1, 6804, 4) --> (1, 6804, 4) --> (1, 6804, 1)
        # all 方法，如果最后一个维度都为 True 则返回 True
        # 筛出去太靠边、尺寸太小的 anchor
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        # (1, 6804, 4) 将介于(0, 1)，的数值映射到实数范围
        anchors = torch.log(anchors / (1 - anchors))
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        # (1, 6804, 4) ~ (1, 6804, 1) --> (1, 6804, 4) 将不符合要求的 anchors 的信息置为 INF 无穷，符合要求则不变
        # True  [x, y, w, h] --> [x, y, w, h]
        # False [x, y, w, h] --> [INF, INF, INF, INF]
        anchors = torch.where(valid_mask, anchors, torch.inf)

        # (1, 6804, 4), (1, 6804, 1)
        return anchors, valid_mask


    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        # input (2, 6804, 256), [(72, 72), (36, 36), (18, 18)], (2, 200, 256), (2, 200, 4)
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            # (1, 6804, 4), (1, 6804, 1)
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)

        # (2, 6804, 256) * (1, 6804, 1) --> (2, 6804, 256) 筛选出 memory 中的信息，有些信息不用预测 anchor，初始化状态这部分被置为 0
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export 
        # (2, 6804, 256) --> (2, 6804, 256)
        output_memory = self.enc_output(memory)

        # (2, 6804, 256) --> (2, 6804, 80)
        enc_outputs_class = self.enc_score_head(output_memory)
        # (2, 6804, 256) --> (2, 6804, 4) + (2, 6804, 4) --> (2, 6804, 4)
        # anchors relative position + anchors = anchor 偏移量 + anchor 原始信息 => 预测后的 anchor
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        # 获取 top300 对应的索引
        # (2, 6804, 80) --> (2, 6804) --> (2, 300)
        # _, topk_ind 两者形状均为 (2, 300)，前者表示筛选出的数值，后者表示筛选出的数值在第一维度（6804）中的索引
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)

        # 获取 top300 对应的 anchor points
        # (2, 300) --> (2, 300, 1) --> (2, 300, 4)
        # (2, 6804, 4) gather (2, 300, 4) --> (2, 300, 4)
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))

        # (2, 300, 4) --> (2, 300, 4) 用于计算损失，而不作为 decoder 的输入
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            # 预测 bbox 信息与加噪 bbox 信息融合
            # (2, 200, 4) concat (2, 300, 4) --> (2, 500, 4)
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        # 获取 top300 对应的类
        # (2, 6804, 80) --> (2, 300, 80) 同样用于计算损失，而不作为 decoder 的输入
        enc_topk_logits = enc_outputs_class.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

        # extract region features
        if self.learnt_init_query: # False 初始值
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            # (2, 6804, 256) --> (2, 300, 256)
            target = output_memory.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            # 预测类别信息与加噪类别融合
            # (2, 200, 256) concat (2, 300, 256) --> (2, 500, 256)
            target = torch.concat([denoising_class, target], 1)
        # (2, 500, 256), (2, 500, 4), (2, 300, 4), (2, 300, 80)
        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits


    def forward(self, feats, targets=None):

        # input projection and embedding
        # (2, 6804, 256)
        # [(72, 72), (36, 36), (18, 18)]
        # [0, 5184, 6480]
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)
        
        # prepare denoising training
        # 将对 gt 加噪后的数据作为 decoder 的输入，让 decoder 学习去噪，使得预测的输出结果逼近真实的 gt
        # 去噪过程是 denoising part，二分部图匹配的部分则是 matching part
        # 对于 denoising part ，我们可以把它看作是一条 shortcut，它可以绕过二分匹配（不去做二分匹配），帮助 decoder 快速的、高效的优化参数
        if self.training and self.num_denoising > 0:
            # (2, 200, 256), (2, 200, 4), (500, 500), {tuple((20,), (100,)), 20, [200, 300]}
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes,  # COCO 数据集类别 80
                    self.num_queries,  # 每张图片多少个 object-query 300
                    self.denoising_class_embed,  # 每种类的编码长度 Embedding(81, 256)
                    num_denoising=self.num_denoising,  # 对于 gt 加噪出多少个正样本和多少个负样本 100
                    label_noise_ratio=self.label_noise_ratio,  # 改变加噪的 gt 中需要改变类别的 gt 的占比 || 类别加噪 0.5
                    box_noise_scale=self.box_noise_scale, )  # bbox 尺度的缩放系数 1
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        # IOU-aware query-selection，从 encoder 编码后的 memory 中筛选出具有先验信息的 query 作为 decoder 的 query 端输入
        # input (2, 6804, 256), [(72, 72), (36, 36), (18, 18)], (2, 200, 256), (2, 200, 4)
        # output (2, 500, 256), (2, 500, 4), (2, 300, 4), (2, 300, 80)
        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        # (6, 2, 500, 4), (6, 2, 500, 80)
        out_bboxes, out_logits = self.decoder(
            target,  # (2, 500, 256)
            init_ref_points_unact,  # (2, 500, 4)
            memory,  # (2, 6804, 256)
            spatial_shapes,  # [(72, 72), (36, 36), (18, 18)]
            level_start_index,  # [0, 5184, 6480]
            self.dec_bbox_head,  # decoder 预测出最终的 bbox，每个 decoder 都有一个预测头 head，输出 (2, 500, 4)
            self.dec_score_head,  # decoder 预测出最终的 cls，每个 decoder 都有一个预测头 head，输出 (2, 500, 80)
            self.query_pos_head,  # 将 anchors 先验信息转化为能与 tgt 对齐融合的 embedding (2, 500, 4) --> (2, 500, 256)
            attn_mask=attn_mask)  # (500, 500)

        # 分离 6 个 decoder 输出的 denosing part 和 matching part 部分
        if self.training and dn_meta is not None:
            # (6, 2, 500, 4) ~ [200, 300] --> (6, 2, 200, 4), (6, 2, 300, 4)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            # (6, 2, 500, 80) ~ [200, 300] --> (6, 2, 200, 80), (6, 2, 300, 80)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)

        # 取出最后一层 decoder 的输出
        # out: {'pred_logits': (2, 300 4), 'pre_boxes': (2, 300, 80)}
        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        if self.training and self.aux_loss:
            # 取出前 5 个 decoder 层的类别输出和预测框输出
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            # 添加 IOU-aware query selection 部分的类别预测输出和预测框预测输出
            # 'aux_outputs':
            # [{'pred_logits': (2, 300, 80), 'pred_boxes': (2, 300, 4)},  decoder_1
            #  {'pred_logits': (2, 300, 80), 'pred_boxes': (2, 300, 4)},  decoder_2
            #  {'pred_logits': (2, 300, 80), 'pred_boxes': (2, 300, 4)},  decoder_3
            #  {'pred_logits': (2, 300, 80), 'pred_boxes': (2, 300, 4)},  decoder_4
            #  {'pred_logits': (2, 300, 80), 'pred_boxes': (2, 300, 4)},  decoder_5
            #  {'pred_logits': (2, 300, 80), 'pred_boxes': (2, 300, 4)}]  IOU-aware query selection
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))
            
            if self.training and dn_meta is not None:
                # 'dn_aux_outputs':
                # [{'pred_logits': (2, 200, 80), 'pred_boxes': (2, 200, 4)} * 6]  decoder_0,1,2,3,4,5
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                # 'dn_meta':
                # {'dn_positive_idx': tuple((20,), (100,)),
                #  'dn_num_group': 20,
                #  'dn_num_split': [200, 300]}
                out['dn_meta'] = dn_meta
        # 字典类型 Dict{ ... }
        # 1. matching part 最后一层 decoder 的预测类和预测框输出，两个键值对: 'pred_logits', 'pred_boxes'
        # 2. matching part 中间 5 层 decoder + IOU-aware query selection 两个部分的预测类和预测框输出，
        #    数组保存六个元素，每个元素两个键值对 ['pred_logits', 'pre_boxes'] * (5 + 1)
        # 3. denoising part 所有六个层 decoder 的预测类和预测框输出
        #    数组保存六个元素，每个元素两个键值对 ['pred_logits', 'pred_boxes'] * 6
        # 4. denoising part 相关信息，包含加噪数据中每个标签的下标，加噪数据的分组数，加噪数据与匹配数据的划分
        #    一个字典保存三个键值对: 'dn_meata': {'dn_positive_idx', 'dn_num_group', 'dn_num_split'}
        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # (5, 2, 300, 80), (5, 2, 300, 4) -->
        # [{'pred_logits': (2, 300, 80), 'pred_boxes': (2, 300, 4)},
        #  {'pred_logits': (2, 300, 80), 'pred_boxes': (2, 300, 4)}
        #  {'pred_logits': (2, 300, 80), 'pred_boxes': (2, 300, 4)}
        #  {'pred_logits': (2, 300, 80), 'pred_boxes': (2, 300, 4)}
        #  {'pred_logits': (2, 300, 80), 'pred_boxes': (2, 300, 4)}]
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]
