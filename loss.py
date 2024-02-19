from typing import Tuple
import torch
import torch.nn
import math

def sum_count(x : torch.Tensor) -> Tuple[torch.Tensor, float]:
    return x.float().sum(), math.prod(x.shape)

class LossHead(torch.nn.Module):
    def __init__(self, focal_gamma : float = 2., class_loss_scale : float = 4., coord_loss_scale : float = 0.1, bg_weight = 0.1):
        super().__init__()

        self.focal_gamma                    = focal_gamma
        self.class_loss_scale               = class_loss_scale
        self.coord_loss_scale               = coord_loss_scale
        self.bg_weight                      = bg_weight
        self.logit_regularization_factor    = 0.

    def forward(
            self,
            class_map : torch.Tensor, correction_map : torch.Tensor,
            target_class_map : torch.Tensor, target_correction_map : torch.Tensor,
            stats : dict, stat_prefix : str
        ) -> torch.Tensor:

        class_map       = class_map.float()
        correction_map  = correction_map.float()
        
        _, num_aspect_levels, _, _, _ = target_class_map.shape
        target_class_map = target_class_map.long()

        class_map       = class_map.reshape(*class_map.shape[:1], num_aspect_levels, -1, *class_map.shape[2:])
        correction_map  = correction_map.reshape(*correction_map.shape[:1], num_aspect_levels, -1, *correction_map.shape[2:])

        bg_class, fg_class = torch.unbind(target_class_map, dim = 2)

        log_prob = class_map * 1
        log_prob = torch.log_softmax(log_prob, dim = 2)

        log_prob = torch.gather(
            torch.cat([
                torch.full((*log_prob.shape[:2], 1, *log_prob.shape[3:]), fill_value = -math.inf, dtype = log_prob.dtype, device = log_prob.device),
                log_prob
            ], dim = 2),
            dim = 2,
            index = target_class_map + 1)
        
        unanimous = ((bg_class < 0) & (fg_class > 0))

        # compute classification loss
        allowed_cls_prob = torch.logsumexp(log_prob, dim = 2)
        class_loss = -allowed_cls_prob * torch.pow((1 - torch.exp(allowed_cls_prob)).clamp(0, None), self.focal_gamma)
        class_loss_weight = torch.where(
            fg_class > 0,
            torch.tensor(1, dtype = class_loss.dtype, device = class_loss.device),
            torch.tensor(self.bg_weight, dtype = class_loss.dtype, device = class_loss.device))
        class_loss.mul_(class_loss_weight)
        class_loss = class_loss.sum()

        # when both bg & fg classes are present, bg class is preferred
        mask = (target_class_map >= 0).all(dim = 2)
        strict_class_loss = log_prob[:,:,0][mask]
        strict_class_loss = -strict_class_loss * torch.pow((1 - torch.exp(strict_class_loss)).clamp(0, None), self.focal_gamma)
        strict_class_loss = strict_class_loss.sum()

        # compute coordinates loss
        coord_loss = (correction_map - target_correction_map).square().sum(dim = 2)
        coord_weight = torch.max(
            log_prob.detach()[:, :, 1].exp().type(coord_loss.dtype),
            unanimous.type(coord_loss.dtype) )
        coord_loss *= coord_weight
        coord_loss = coord_loss.sum()
        
        # compute statistics
        _, predicted_class = torch.max(class_map, dim = 2)
        recall_sum, recall_cnt = sum_count(predicted_class[unanimous] == fg_class[unanimous])
        stats[stat_prefix + "-RECALL    "] = recall_sum, recall_cnt
        stats[stat_prefix + "-FPR       "] = ((predicted_class != 0) & (predicted_class != fg_class)).float().sum(), recall_cnt
        stats[stat_prefix + "-COORD-ERR "] = (coord_loss.detach(), coord_weight.sum())
        stats[stat_prefix + "-cls-loss  "] = (class_loss.detach(), 1)
        stats[stat_prefix + "-cls-strict"] = (strict_class_loss.detach(), 1)
        stats[stat_prefix + "-loc-loss  "] = (coord_loss.detach(), 1)

        return (
            (class_loss + strict_class_loss * 0.02) * self.class_loss_scale
            + coord_loss * self.coord_loss_scale
            + class_map.square().mean() * self.logit_regularization_factor)
