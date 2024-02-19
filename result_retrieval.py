import math
from typing import Tuple
import torch

from .metadata import ObjectDetectorMetadata

def overlap_ratio(b1 : torch.Tensor, b2 : torch.Tensor) -> torch.Tensor:
    area1 = b1[..., 2] * b1[..., 3]
    area2 = b2[..., 2] * b2[..., 3]

    iw = ((b1[..., 2] + b2[..., 2]) / 2 - torch.abs(b1[..., 0] - b2[..., 0])).clamp(0, None)
    ih = ((b1[..., 3] + b2[..., 3]) / 2 - torch.abs(b1[..., 1] - b2[..., 1])).clamp(0, None)
    iw = torch.min(iw, torch.min(b1[..., 2], b2[..., 2]))
    ih = torch.min(ih, torch.min(b1[..., 3], b2[..., 3]))

    return iw * ih / torch.min(area1, area2)

class ObjectDetectionResultRetriever:
    def __init__(self, metadata : ObjectDetectorMetadata, device : torch.device, max_overlap : float = 0.8):
        super().__init__()

        self.prior_sizes    = torch.tensor(metadata.calc_prior_box_sizes(), dtype = torch.float, device = device)
        self.metadata       = metadata
        self.max_overlap    = max_overlap
    
    def get_proposals_miplevel(self, logit_map : torch.Tensor, correction_map : torch.Tensor, bg_bias : float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, height, width = logit_map.shape
        logit_map = logit_map.reshape(self.metadata.num_aspect_levels, 1 + len(self.metadata.class_names), height, width) # [aspect_level, class, Y, X]
        correction_map = correction_map.reshape(self.metadata.num_aspect_levels, 4, height, width) # [aspect_level, 4, Y, X]

        logit_map      = logit_map.float()
        correction_map = correction_map.float()

        logit_map[:,0,:,:] = logit_map[:,0,:,:] + bg_bias
        logit_map = torch.log_softmax(logit_map, dim = 1)
        class_prob, class_id = torch.max(logit_map, dim = 1)

        is_valid = (class_id > 0)

        # # apply non-maximum suppression
        # mask1 = class_id  [:-1, :, :] != class_id  [1:, :, :]
        # mask2 = class_prob[:-1, :, :] >  class_prob[1:, :, :]
        # is_valid[ :-1, :, :] &= mask1 |  mask2
        # is_valid[1:  , :, :] &= mask1 | ~mask2
        # 
        # mask1 = class_id  [:, :-1, :] != class_id  [:, 1:, :]
        # mask2 = class_prob[:, :-1, :] >  class_prob[:, 1:, :]
        # is_valid[:,  :-1, :] &= mask1 |  mask2
        # is_valid[:, 1:  , :] &= mask1 | ~mask2
        # 
        # mask1 = class_id  [:, :, :-1] != class_id  [:, :, 1:]
        # mask2 = class_prob[:, :, :-1] >  class_prob[:, :, 1:]
        # is_valid[:, :,  :-1] &= mask1 |  mask2
        # is_valid[:, :, 1:  ] &= mask1 | ~mask2

        # make proposals
        aspect_level, cy, cx = torch.nonzero(is_valid, as_tuple = True)

        class_id    = class_id  [aspect_level, cy, cx]
        class_prob  = class_prob[aspect_level, cy, cx]

        coordinates = torch.cat([
            cx.float()[:,None],
            cy.float()[:,None],
            self.prior_sizes[aspect_level]
        ], dim = 1)
        coordinates += correction_map[aspect_level, :, cy, cx]

        # filter out invalid proposals
        is_valid    = (coordinates[..., 2:4] > 0).all(dim = -1)
        class_id    = class_id[is_valid]
        class_prob  = class_prob[is_valid]
        coordinates = coordinates[is_valid]

        # convert to zero-based class ID
        class_id.sub_(1)

        return class_id, class_prob, coordinates

    def get_proposals(self, maps : list, batch_index : int, bg_bias : float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_class_id      = []
        l_class_prob    = []
        l_coordinates   = []

        for level in range(0, self.metadata.num_mip_levels):
            scale = self.metadata.first_level_scale * math.pow(2, level)
            logit_map, correction_map = maps[level]
            class_id, class_prob, coordinates = self.get_proposals_miplevel(logit_map[batch_index], correction_map[batch_index], bg_bias = bg_bias)

            coordinates *= scale
        
            l_class_id.append(class_id)
            l_class_prob.append(class_prob)
            l_coordinates.append(coordinates)
        
        l_class_id      = torch.cat(l_class_id, dim = 0)
        l_class_prob    = torch.cat(l_class_prob, dim = 0)
        l_coordinates   = torch.cat(l_coordinates, dim = 0)

        return l_class_id, l_class_prob, l_coordinates

    def remove_duplicates(
            self,
            class_id : torch.Tensor, class_prob : torch.Tensor, coordinates : torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        i = torch.arange(0, class_id.size(0), dtype = torch.int, device = class_id.device)
        compare_mask = (class_id[:,None] == class_id[None,:]) & (i[:,None] < i[None,:])

        i, j = torch.nonzero(compare_mask, as_tuple = True)
        m1 = overlap_ratio(coordinates[i], coordinates[j]) >= self.max_overlap
        m2 = class_prob[i] > class_prob[j]

        eliminate_indices = torch.cat([i[m1 & ~m2], j[m1 & m2]], dim = 0)
        keep_mask = torch.ones((class_id.size(0),), dtype = torch.bool, device = class_id.device)
        keep_mask[eliminate_indices] = False

        class_id    = class_id[keep_mask]
        class_prob  = class_prob[keep_mask]
        coordinates = coordinates[keep_mask]

        return class_id, class_prob, coordinates

    def __call__(self, maps : list, batch_index : int, threshold : float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bg_bias = math.log(threshold / (1. - threshold))

        proposals = self.get_proposals(maps, batch_index, bg_bias = bg_bias)
        proposals = self.remove_duplicates(*proposals)
        return proposals
