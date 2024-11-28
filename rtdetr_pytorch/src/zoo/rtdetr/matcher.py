"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

by lyuwenyu
"""

import torch
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from src.core import register


@register
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # 2， 300 <--{'pred_logits': (2, 300, 80), 'pred_boxes': (2, 300, 4), 'dn_meta'{ ... }}
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss: # 默认 False，但配置文件 True
            # (2, 300, 80) --> (600, 80) --> (600, 80)
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        # (2, 300, 4) --> (600, 4)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        # (6,) <-- 1 + 5 类别: [23, 56, 0, 52, 0, 0]
        tgt_ids = torch.cat([v["labels"] for v in targets])
        # (6, 4)
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            # 构建损失矩阵
            # (600, 80) --> (600, 6)
            out_prob = out_prob[:, tgt_ids]
            # 此处不知道每个 bbox 的真实标签，因此无法区分正负样本类型，因此对于每个 bbox 既计算出它作为正样本的损失，又计算出它作为负样本的损失
            # 然后，将它作为正样本的损失减去作为负样本的损失，得到最终的损失值
            # Focal Loss
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            # (600, 6)
            cost_class = pos_cost_class - neg_cost_class        
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        # (600, 4) ~ (6, 4) --> (600, 6)
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        # (600, 4) ~ (6, 4) --> (600, 6)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        # 损失值 = 类别损失 + 定位损失 = 2 * focal loss + 5 * L1 loss + 2 * GIOU loss
        # (600, 6)
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # (2, 300, 6)
        C = C.view(bs, num_queries, -1).cpu()

        # [1, 5]
        sizes = [len(v["boxes"]) for v in targets]
        # 匈牙利算法 || from scipy.optimize import linear_sum_assignment
        # (2, 300, 6) ~ [1, 5] --> [(2, 300, 1), (2, 300,5)]
        # [tuple((1,), (1,)), tuple((5,), (5,))]，tuple 包含的元素为列表类型，后面需要转化为 Tensor 类型
        # [([218], [0]), ([19, 161, 228, 250, 256], [3, 0, 1, 2, 4])]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        # 将结果转换为 tensor，形状不变
        # [tuple((1,), (1,)), tuple((5,), (5,))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
