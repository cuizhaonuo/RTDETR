"""by lyuwenyu
"""

import torch 

from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh



def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0,):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None

    # [1, 5] 每张图片中标签的数量
    num_gts = [len(t['labels']) for t in targets]
    device = targets[0]['labels'].device

    # 5 本次 batch 中图像的最多标签数
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    # 20 = 100 / 5
    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts)

    # 初始化
    # (2, 5) 填充 80，因为 COCO 数据集有 80 个类别
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    # (2, 5, 4) 填充 0
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    # (2, 5) 填充 0
    # 0 0 0 0 0
    # 0 0 0 0 0
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    # 将 gt 的相关信息填充进初始化的变量中
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            # [0, :1] ~ (1,)
            # [1, :5] ~ (5,)
            # 一维填充一维
            input_query_class[i, :num_gt] = targets[i]['labels']
            # [0, :1] = [0, :1, :] ~ (1, 4)
            # [1, :5] = [1, :5, :] ~ (5, 4)
            # 二维填充二维
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            # [0, :1] ~ 1
            # [1, :5] ~ 1
            # 1 0 0 0 0
            # 1 1 1 1 1
            pad_gt_mask[i, :num_gt] = 1

    # each group has positive and negative queries.
    # 重复平铺以构建正负样本的初始化状态
    # (2, 5) --> (2, 5*2*20) => (2, 200)
    input_query_class = input_query_class.tile([1, 2 * num_group])
    # (2, 5, 4) --> (2, 200, 4)
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    # (2, 5) --> (2, 200)
    # 1 0 0 0 0 | 1 0 0 0 0  * 20
    # 1 1 1 1 1 | 1 1 1 1 1  * 20
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    # (2, 5*2, 1) => (2, 10, 1) 填充 0
    # 0 0 0 0 0 | 0 0 0 0 0
    # 0 0 0 0 0 | 0 0 0 0 0
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    # (2, 5:) = (2, 5:, :) 填充 1，此时全部的负样本均填充为了 1，而全部正样本则为 0
    # 0 0 0 0 0 | 1 1 1 1 1
    # 0 0 0 0 0 | 1 1 1 1 1
    negative_gt_mask[:, max_gt_num:] = 1
    # (2, 10*20, 1) => (2, 200, 1)
    # 0 0 0 0 0 | 1 1 1 1 1  * 20
    # 0 0 0 0 0 | 1 1 1 1 1  * 20
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    # 获正样本的 mask，反转正负样本的数值，此时全部负样本转换为了 0，而全部正样本则为 1
    # 1 1 1 1 1 | 0 0 0 0 0  * 20
    # 1 1 1 1 1 | 0 0 0 0 0  * 20
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    # (2, 200, 1) --> (2, 200) * (2, 200) --> (2, 200)
    # 正样本标签处为 1，其他均为 0
    # 1 0 0 0 0 | 0 0 0 0 0  * 20
    # 1 1 1 1 1 | 0 0 0 0 0  * 20
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    # 获取非零元素的索引，即标签的索引
    # (20+100, 0) => (120, 0): (0, 10, 20, ..., 0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, ...)
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    # (120, 0)  [20, 100] --> tuple((20,), (100,))
    # tuple((0, 10, 20, ...), (0, 1, 2, 3, 4, ...]) 元素均为 tensor 类型
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    # 200
    # num_denoising // max_gt_num 若 max_gt_num 为奇数，分组数是向下取整后的数
    # 100 // 7 => 14, 7 * 2 * 14 = 196 < 200
    num_denoising = int(max_gt_num * 2 * num_group)

    # 类别加噪
    if label_noise_ratio > 0:
        # (2, 200) int32 --> (2, 200) bool 生成的随机数范围 [0, 1)，其中小于 0.25 的元素置为 True，此即待加噪的类标签，其他为 False
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        # (2, 200) bool --> (2, 200) int32 生成的随机数范围 [0, 80)，注意并不是只有 True 才填入随机数，而是全部位置都填入随机数
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        # 举一个包含正负样本的组的例子
        # 0 1 0 0 1 | 1 0 0 0 1  &  1 0 0 0 0 | 1 0 0 0 0  =  0 0 0 0 0 | 1 0 0 0 0  ==> 80 80 80 80 80 | 78`80 80 80 80 80
        # 1 0 1 1 0 | 1 1 0 0 0  &  1 1 1 1 1 | 1 1 1 1 1     1 0 1 1 0 | 1 1 0 0 0      36` 4 47`65` 9 | 54`42` 7  8  8  9
        # 在最后计算的矩阵中，True 填入对应位置新生成的标签，False 填入对应位置原始标签，完成加噪过程
        # (2, 200)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    # if label_noise_ratio > 0:
    #     input_query_class = input_query_class.flatten()
    #     pad_gt_mask = pad_gt_mask.flatten()
    #     # half of bbox prob
    #     # mask = torch.rand(input_query_class.shape, device=device) < (label_noise_ratio * 0.5)
    #     mask = torch.rand_like(input_query_class) < (label_noise_ratio * 0.5)
    #     chosen_idx = torch.nonzero(mask * pad_gt_mask).squeeze(-1)
    #     # randomly put a new one here
    #     new_label = torch.randint_like(chosen_idx, 0, num_classes, dtype=input_query_class.dtype)
    #     # input_query_class.scatter_(dim=0, index=chosen_idx, value=new_label)
    #     input_query_class[chosen_idx] = new_label
    #     input_query_class = input_query_class.reshape(bs, num_denoising)
    #     pad_gt_mask = pad_gt_mask.reshape(bs, num_denoising)

    # 坐标加噪
    if box_noise_scale > 0:
        # (2, 200, 4) --> (2, 200, 4)
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        # (2, 200, 4) --> (2, 200, 2) [w, h] --> (2, 200, 2) [w/2, h/2] --> (2, 200, 4) [w/2, h/2, w/2, h/2]
        # 4d  0  0  0  0 | 4d  0  0  0  0
        # 4d 4d 4d 4d 4d | 4d 4d 4d 4d 4d
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        # 模拟随机正负号
        # (2, 200, 4) --> (2, 200, 4) int[0, 1] --> (2, 200, 4) int[-1, 1] 在上一步中随机数为 1 的值不变，但是其他值变为 -1
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        # (2, 200, 4) --> (2, 200, 4) [0, 1)
        rand_part = torch.rand_like(input_query_bbox)
        # rand_part + 1.0 表示分布从 [0, 1) 转换为 [1, 2)
        # 负样本所有位置填充 [1, 2) 分布，正样本所有位置填充 [0, 1) 分布
        # 注意：这里只是举一个例子，实际的元素是一个四维的 xyxy，而不是如下的单个数值
        # 负样本
        # ... | 1.6968, 1.0633, 1.8247, 1.0008, 1.5444  *  0 0 0 0 0 | 1 1 1 1 1
        # ... | 1.2737, 1.3383, 1.2672, 1.2934, 1.2217  *  0 0 0 0 0 | 1 1 1 1 1
        # 正样本
        # 0.2866, 0.3553, 0.1537, 0.7985, 0.0813 | ...  *  1 1 1 1 1 | 0 0 0 0 0
        # 0.0100, 0.4882, 0.8336, 0.3138, 0.2810 | ...  *  1 1 1 1 1 | 0 0 0 0 0
        # (2, 200, 4) [0, 1) --> (2, 200, 4) [1, 2) * (2, 200, 4) bool --> (2, 200, 4)
        # (2, 200, 4) [0, 1) * (2, 200, 4) bool --> (2, 200, 4)
        # (2, 200, 4) = (2, 200, 4) + (2, 200, 4) 负样本 + 正样本
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        # (2, 200, 4) 给缩放系数添加正负号
        rand_part *= rand_sign
        # (2, 200, 4) [λ1 * w/2, λ2 * h/2, λ3 * w/2, λ4 * h/2]具备正负号的缩放系数与 [w/2, h/2, w/2, h/2]
        # known_bbox (2, 200, 4)
        # bb  0  0  0  0 | bb  0  0  0  0  +  0.8074,  0.0      0.0,    -0.0,     0.0    | 1.8074,  0.0      0.0,    -0.0,     0.0
        # bb bb bb bb bb | bb bb bb bb bb     0.2092, -0.5912, -0.6738,  0.7284, -0.1980 | 1.2092, -1.5912, -1.6738,  1.7284, -1.1980
        #
        # -0.2712,  0.0      0.0,    -0.0,     0.0    | -1.2712,  0.0      0.0,    -0.0,     0.0
        #  0.8403,  1.1181,  1.4370, -0.3995,  0.2495 |  1.8403,  2.1181,  2.4370, -1.3995, -0.7505
        known_bbox += rand_part * diff
        #  0.0,    0.0  0.0, 0.0, 0.0    | 0.0, 0.0, 0.0, 0.0, 0.0
        #  0.8403, 1.0, 1.0, 0.0, 0.2495 | 1.0, 1.0, 1.0, 0.0, 0.0
        known_bbox.clip_(min=0.0, max=1.0)
        # (2, 200, 4) xyxy --> (2, 200, 4) cxcywh
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        # inverse_sigmoid 函数（或称 logit 函数）将 (0, 1) 区间的概率值映射回实数范围，恢复初始数值大小。
        # (2, 200, 4) --> (2, 200, 4) 相对值变为绝对值
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    # class_embed = torch.concat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=device)])
    # input_query_class = torch.gather(
    #     class_embed, input_query_class.flatten(),
    #     axis=0).reshape(bs, num_denoising, -1)
    # input_query_class = class_embed(input_query_class.flatten()).reshape(bs, num_denoising, -1)
    # (2, 200) --> (2, 200, 256)
    input_query_class = class_embed(input_query_class)

    # 500 = 200 + 300
    tgt_size = num_denoising + num_queries
    # attn_mask = torch.ones([tgt_size, tgt_size], device=device) < 0
    # (500, 500) bool 填充 False
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    # (500, 500) (100:500, 0:100) --> (500, 500)
    # 在 attn_mask 中不可见为 True，可见为 False
    attn_mask[num_denoising:, :num_denoising] = True

    # reconstruct cannot see each other
    # range[0, 1, 2, ... 19]，每个组有 10 个样本
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
        
    dn_meta = {
        # tuple((20,), (100,))
        "dn_positive_idx": dn_positive_idx,
        # 20 = num_noising // max_gt_num = 100 // 5
        "dn_num_group": num_group,
        # [200, 300]，100 组加噪数据，每组数据包含 5 个正样本 5 个负样本
        "dn_num_split": [num_denoising, num_queries]
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])

    # (2, 200, 256), (2, 200, 4), (500, 500), {tuple((20,), (100,)), 20, [200, 300]}
    return input_query_class, input_query_bbox, attn_mask, dn_meta
