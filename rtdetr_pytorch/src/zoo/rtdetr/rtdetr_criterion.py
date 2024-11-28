"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""


import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision

# from torchvision.ops import box_convert, generalized_box_iou
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.core import register



@register
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        # ce_loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction="none")
        # prob = F.sigmoid(src_logits) # TODO .detach()
        # p_t = prob * target + (1 - prob) * (1 - target)
        # alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        # loss = alpha_t * ce_loss * ((1 - p_t) ** self.gamma)
        # loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        # indices:
        # [([218], [0]), ([19, 161, 228, 250, 256], [3, 0, 1, 2, 4])]
        # 218 -> 0
        # 19, 161, 228, 250, 256 -> 3, 0, 1, 2, 4
        # matcher 根据匈牙利匹配从 300 个预测中筛选出作为最终结果的预测，获取这些预测的索引
        # ([0, 1, 1, 1, 1, 1], [218, 19, 161, 228, 250, 256])
        idx = self._get_src_permutation_idx(indices)

        # 获取 src 对应的预测 bbox
        # (2, 300, 4) ~ ([0, 1, 1, 1, 1, 1], [218, 19, 161, 228, 250, 256]) --> (6, 4)
        # 第一个列表 [0, 1, 1, 1, 1, 1] 是 batch 索引。
        # 第二个列表 [218, 19, 161, 228, 250, 256] 是 box 索引。
        src_boxes = outputs['pred_boxes'][idx]
        # 获取标签 bbox
        # (1, 4) ~ [0] --> (4,)
        # (5, 4) ~ [3, 0, 1, 2, 4] --> (5, 4)
        # (4,) concat (5, 4) --> (6, 4)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # 计算 GIOU
        # (6, 4) ~ (6, 4) --> (6, 6)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        # 取对角线的值，即符合匈牙利匹配的 iou
        # (6, 6) --> (6,)
        ious = torch.diag(ious).detach()

        # (2, 300, 80)
        src_logits = outputs['pred_logits']
        # 获取标签 class
        # (1,) ~ [0] --> (1,) : 31
        # (5,) ~ [3, 0, 1, 2, 4] --> (5,) : 45, 57, 26, 64, 31
        # (1,) concat (5,) --> (6,)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # (2, 300) 填充 80
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # (2, 300) ~ ([0, 1, 1, 1, 1, 1], [218, 19, 161, 228, 250, 256]) 对应位置填充真实标签 (6,) --> (2, 300)
        # 第一行 300 在 218 处填写 31
        # 第二行 300 在 218, 19, 161, 228, 250, 256 处分别填写 45, 57, 26, 64, 31
        target_classes[idx] = target_classes_o
        # (2, 300) --> (2, 300, 81) --> (2, 300, 80)
        # 例子
        # [[80, 80, 80, ...  52, 80, 80]
        #  [80, 80, 80, ... 80, 80, 80]]
        # one-hot 编码
        # [[[0, 0, 0, ... 0, 0, 1]
        #   [0, 0, 0, ... 0, 0, 1]
        #   ...
        #   [0, 0, 0, ... 1, 0, 0]
        #   [0, 0, 0, ... 0, 0, 1]
        #   [0, 0, 0, ... 0, 0, 1]]
        #  [[[0, 0, 0, ... 0, 0, 1]
        #    ...
        #    [0, 0, 0, ... 0, 0, 1]]]
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        # (2, 300) 填充 0
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        # (2, 300) ~ ([0, 1, 1, 1, 1, 1], [218, 19, 161, 228, 250, 256]) 填充 ious (6,) --> (2, 300)
        # [[0, 0, 0, ..., iou, 0, 0]
        #  [0, 0, 0, ...,   0, 0, 0]]
        target_score_o[idx] = ious.to(target_score_o.dtype)
        # (2, 300) --> (2, 300, 1) * (2, 300, 80) --> (2, 300, 80)
        # 将 one-hot 对应的 1 转变为 iou
        # [[[0, 0, 0, ... 0, 0, 1]
        #   [0, 0, 0, ... 0, 0, 1]
        #   ...
        #   [0, 0, 0, ... iou, 0, 0]
        #   [0, 0, 0, ...   0, 0, 1]
        #   [0, 0, 0, ...   0, 0, 1]]
        #  [[[0, 0, 0, ... 0, 0, 1]
        #    ...
        #    [0, 0, 0, ... 0, 0, 1]]]
        target_score = target_score_o.unsqueeze(-1) * target

        # src_logits: (2, 300, 80)，网络输出的结果，每个预测框 80 个类别的预测置信度
        # pred_score: (2, 300, 80)，对应于 src_logits，是经过 sigmiod 处理后的 src_logits
        # target_score: (2, 300, 80)，匈牙利匹配的预测框和 GT，数值为 IOU，其他值为 0，24000 里面只有 6 个具有 IOU 数值
        # target: (2, 300, 80)，one-hot 编码，对应于 target_score，至少具有填充 80 的背景值

        # 对预测结果进行 sigmoid
        # (2, 300, 80) --> (2, 300, 80)
        pred_score = F.sigmoid(src_logits).detach()
        # 0.2 * (2, 300, 80)^2.0 * (1 - (2, 300, 80) one-hot) + (2, 300, 80)
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        # (2, 300, 80)
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        # (2, 300, 80) --> (2, 80) --> (1,) * 300 / 6 --> (1,)
        # (2, 80) 每张图片中对所有预测框的类别损失的平均值，
        # 将对 (2, 80) 中的所有元素求和，使结果为一个标量。这是为了将损失归约到一个值，方便优化器更新模型参数。
        # 在计算过程中，每张图片的 300 个框被归约为一个均值，因此这里通过乘以 300 将框的数量重新考虑进来，恢复到每张图片的总框损失。
        # 在分布式训练或有批次的情况下，num_boxes 是所有批次中真实框的总数。
        # 乘以 src_logits.shape[1]：恢复框的数量影响。
        # 除以 num_boxes：标准化损失，使其与真实框的数量成比例，避免因批次中预测框数量不同导致的不均衡。
        # 0.1594 一个最终的 loss 值
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # ([0, 1, 1, 1, 1, 1], [218, 19, 161, 228, 250, 256])
        idx = self._get_src_permutation_idx(indices)
        # (2, 300, 4) ~ ([0, 1, 1, 1, 1, 1], [218, 19, 161, 228, 250, 256]) --> (6, 4)
        src_boxes = outputs['pred_boxes'][idx]
        # indices:
        # [([218], [0]), ([19, 161, 228, 250, 256], [3, 0, 1, 2, 4])]
        # 218 -> 0
        # 19, 161, 228, 250, 256 -> 3, 0, 1, 2, 4
        # (1, 4) concat (5, 4) --> (6, 4)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        # (6, 4) ~ (6, 4) --> (6, 4)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        # (6, 4) --> (1,) / 6 --> (1,): 0.2431
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # (6, 4) ~ (6, 4) --> (6, 6) --> (6,)
        # 1 - (6,) --> (6,)
        loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
        # (6,) --> (1,) / 6 --> (1,): 0.7517
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        # {'loss_bbox': 0.2431, 'loss_giou': 0.7517}
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        # indices:
        # [([218], [0]), ([19, 161, 228, 250, 256], [3, 0, 1, 2, 4])]
        # 218 -> 0
        # 19, 161, 228, 250, 256 -> 3, 0, 1, 2, 4
        # 为标签划分 batch，每张图片所属的标签属于同一 batch
        # [0, 1, 1, 1, 1, 1]
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # 拼接所有 src 的索引
        # [218, 19, 161, 228, 250, 256]
        src_idx = torch.cat([src for (src, _) in indices])
        # [0, 1, 1, 1, 1, 1], [218, 19, 161, 228, 250, 256]
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,

            'bce': self.loss_labels_bce,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # output Dict{ ... }
        # 1. (2, 300, 80) 'pre_logits'
        # 2. (2, 300, 4)  'pre_boxes'
        # 3, [{(2, 300, 80), (2, 300, 4)} * 6]  'aux_outputs'
        # 4, [{(2, 200, 80), (2, 200, 4)} * 6]  'dn_aux_outputs'
        # 5, {tuple((20,), (100,)), 20, [200, 300]}  'dn_meta'
        #
        # target Dict{ ...}
        # {{'boxes': (4,), 'labels': (1,), 'image_id', 'area', 'iscrowd', 'orig_size', 'size'}
        #  {'boxes': (5, 4), 'labels': (5,), 'image_id', 'area', 'iscrowd', 'orig_size', 'size'}
        #
        # 取出最后一层的预测类和预测框输出，和 dn_meta（这部分不发挥作用）
        # {'pred_logits': (2, 300, 80), 'pred_boxes': (2, 300, 4), 'dn_meta'{ ... }}
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        # [tuple((1,), (1,)), tuple((5,), (5,))]
        # [([218], [0]), ([19, 161, 228, 250, 256], [3, 0, 1, 2, 4])]
        # 218 -> 0
        # 19, 161, 228, 250, 256 -> 3, 0, 1, 2, 4
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # 6 = 1 + 5
        num_boxes = sum(len(t["labels"]) for t in targets)
        # 6 --> tensor 6
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        # 6 / 1 --> 6 单 cpu
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        # ['vfl', 'boxes']
        for loss in self.losses:
            # {'loss_vfl': 0.1594}
            # {'loss_bbox': 0.2431, 'loss_giou': 0.7517}
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            # {loss_vfl: 1, loss_bbox: 5, loss_giou: 2}
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)
        # losses: {'loss_vfl': 0.1594 * 1, 'loss_bbox': 0.2431 * 5, 'loss_giou': 0.7517 * 2}

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # [{(2, 300, 80), (2, 300, 4)} * 6]  'aux_outputs'
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # {'pred_logits': (2, 300, 80), 'pred_boxes': (2, 300, 4)}
                # {(2, 300, 80), (2, 300, 80)}, {(2,), (3,)} -->
                # indices 每个批次都不一样
                # indices:
                # [([100], [0]), ([10, 30, 20, 200, 100], [4, 3, 2, 1, 0])]
                # 100 -> 0
                # 10, 30, 20, 200, 100 -> 4, 3, 2, 1, 0
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    # 修改名字
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        # losses: len 3 + 6 * 3 == 21
        # {'loss_vfl': 0.1594 * 1, 'loss_bbox': 0.2431 * 5, 'loss_giou': 0.7517 * 2,
        #  'loss_vfl_aux_0': 0.1573, 'loss_bbox_aux_0': 3.0829, ‘loss_giou_aux_0’: 1.7769,
        #  'loss_vfl_aux_1': 0.1673, 'loss_bbox_aux_1': 3.1453, ‘loss_giou_aux_1’: 1.6789,
        #  ... ...
        #  'loss_vfl_aux_5': 0.1642, 'loss_bbox_aux_5': 3.2411, ‘loss_giou_aux_5’: 1.7435 }

        # In case of cdn auxiliary losses. For rtdetr
        # [{(2, 200, 80), (2, 200, 4)} * 6]  'dn_aux_outputs'
        # {tuple((20,), (100,)), 20, [200, 300]}  'dn_meta'
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            # {tuple((20,), (100,)), 20, [200, 300]}  outputs['dn_meta']
            # {{'boxes': (4,), 'labels': (1,) ... }, {'boxes': (5, 4), 'labels': (5,) ... } targets
            # indices:
            # [((20,), (20,)), ((100,), (100,))]
            # [([0, 10, ... 180, 190], [0, 0, ... 0, 0]),
            #  ([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, ...],
            #   [0, 1, 2, 3, 4,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4, ...])]
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            # 120 = 6 * 20
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        # losses: 3 + 6 * 3 + 6 * 3 == 39
        # {'loss_vfl': 0.1594 * 1, 'loss_bbox': 0.2431 * 5, 'loss_giou': 0.7517 * 2,
        #  // aux_outputs
        #  'loss_vfl_aux_0': 0.1573, 'loss_bbox_aux_0': 3.0829, 'loss_giou_aux_0': 1.7769,
        #  'loss_vfl_aux_1': 0.1673, 'loss_bbox_aux_1': 3.1453, 'loss_giou_aux_1': 1.6789,
        #  ... ...
        #  'loss_vfl_aux_5': 0.1642, 'loss_bbox_aux_5': 3.2411, 'loss_giou_aux_5': 1.7435
        #  // dn_aux_outputs
        #  'loss_vfl_dn_0': 0.1467, 'loss_bbox_dn_0': 3.3457, 'loss_giou_dn_0': 1.6431,
        #  'loss_vfl_dn_0': 0.1368, 'loss_bbox_dn_0': 3.234, 'loss_giou_dn_0': 1.123,
        #  ... ...
        #  'loss_vfl_dn_0': 0.1256, 'loss_bbox_dn_0': 3.853, 'loss_giou_dn_0': 1.986 } <- END
        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        '''get_cdn_matched_indices
        '''
        # outputs['dn_meta']:
        # {'dn_positive_idx': ((20,), (100,)),
        #  'dn_num_group': 20,
        #  'dn_num_split': [200, 300]}
        # targets
        # {{'boxes': (4,), 'labels': (1,) ... }, {'boxes': (5, 4), 'labels': (5,) ... }}

        # ((20,), (100,)), 20
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        # [1, 5]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        # 1, 5 in [1, 5]
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                # [0]
                # [0, 1, 2, 3, 4]
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                # [0, 0, ... 0] len: 20 = 1 * 20
                # [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ... 0, 1, 2, 3, 4] len: 100 = 5 * 20
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                # [((20,), (20,)]
                # [((20,), (20,)), ((100,), (100,))]
                # [([0, 10, ... 180, 190], [0, 0, ... 0, 0]),
                #  ([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, ...],
                #   [0, 1, 2, 3, 4,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4, ...])]
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices





@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




