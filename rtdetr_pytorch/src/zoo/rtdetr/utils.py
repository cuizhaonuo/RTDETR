"""by lyuwenyu
"""

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F 


def inverse_sigmoid(x: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    # input (2, 6804, 8, 32), [[72, 72], [36, 36], [18, 18]], (2, 500, 8, 3, 4, 2), (2, 500, 8, 3, 4)
    # 2, 8, 32 <-- (2, 6804, 8, 32)
    bs, _, n_head, c = value.shape
    # 500, 3, 4 <-- (2, 500, 8, 3, 4, 2)
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    # [5184, 1296, 324]
    split_shape = [h * w for h, w in value_spatial_shapes]
    # (2, 5184, 8, 32)
    # (2, 1296, 8, 32)
    # (2, 324, 8, 32)
    # [(2, 5184, 8, 32), (2, 1296, 8, 32), (2, 324, 8, 32)]
    value_list = value.split(split_shape, dim=1)
    # 定义采样点，归一化坐标在 [-1, 1] 范围内
    # 将左上角 (0,0)，右下角 (1, 1) 的描述方式，转变为中心 (0, 0)，左上角 (-1, -1)，右下角 (1, 1) 的描述方式
    # (2, 500, 8, 3, 4, 2) --> (2, 500, 8, 3, 4, 2)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        # (2, 5184, 8, 32) --> (2, 5184, 256) --> (2, 256, 5184) --> (16(2*8), 32, 72, 72)
        # (2, 1296, 8, 32) --> (2, 1296, 256) --> (2, 256, 1296) --> (16, 32, 36, 36)
        # (2, 324, 8, 32) --> (2, 324, 256) --> (2, 256, 324) --> (16, 32, 18, 18)
        value_l_ = value_list[level].flatten(2).permute(
            0, 2, 1).reshape(bs * n_head, c, h, w)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        # (2, 500, 8, 3, 4, 2) --> (2, 500, 8, 4, 2) --> (2, 8, 500, 4, 2) --> (16, 500, 4, 2)
        # (2, 500, 8, 3, 4, 2) --> (2, 500, 8, 4, 2) --> (2, 8, 500, 4, 2) --> (16, 500, 4, 2)
        # (2, 500, 8, 3, 4, 2) --> (2, 500, 8, 4, 2) --> (2, 8, 500, 4, 2) --> (16, 500, 4, 2)
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(
            0, 2, 1, 3, 4).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        # 取 sample_locations 对应位置的 value，(N, C, H_in, W_in) ~ (N, H_out, W_out, 2) --> (N, C H_out, W_out)
        # (16, 32, 72, 72) ~ (16, 500, 4, 2) level_0 --> (16, 32, 500, 4)
        # (16, 32, 36, 36) ~ (16, 500, 4, 2) level_1 --> (16, 32, 500, 4)
        # (16, 32, 18, 18) ~ (16, 500, 4, 2) level_2 --> (16, 32, 500, 4)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        # [(16, 32, 500, 4)]
        # [(16, 32, 500, 4), (16, 32, 500, 4)]
        # [(16, 32, 500, 4), (16, 32, 500, 4), (16, 32, 500, 4)]
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    # (2, 500, 8, 3, 4) --> (2, 8, 500, 3, 4) --> (16, 1, 500, 12)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points)

    # [(16, 32, 500, 4), (16, 32, 500, 4), (16, 32, 500, 4)] --> (16, 32, 500, 3, 4)
    # (16, 32, 500, 3, 4) --> (16, 32, 500, 12) * (16, 1, 500, 12) --> (16, 32, 500, 12) --> (16, 32, 500) --> (2, 256, 500)
    output = (torch.stack(
        sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

    # (2, 256, 500) --> (2, 500, 256)
    return output.permute(0, 2, 1)


import math 
def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init



def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 


