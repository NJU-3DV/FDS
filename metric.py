"""Metrics"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class DepthMetrics(nn.Module):
    """Computation of error metrics between predicted and ground truth depths

    from:
        https://arxiv.org/abs/1806.01260

    Returns:
        abs_rel: normalized avg absolute realtive error
        sqrt_rel: normalized square-root of absolute error
        rmse: root mean square error
        rmse_log: root mean square error in log space
        a1, a2, a3: metrics
    """

    def __init__(self, tolerance: float = 0.1, **kwargs):
        self.tolerance = tolerance
        super().__init__()

    @torch.no_grad()
    def forward(self, pred, gt):
        if pred.shape != gt.shape:
            pred = pred.unsqueeze(0).unsqueeze(0)
            pred = torch.nn.functional.interpolate(pred, size=gt.shape[-2:], mode='bilinear', align_corners=False).squeeze()
            gt = gt.squeeze()
        mask = gt > self.tolerance

        thresh = torch.max((gt[mask] / pred[mask]), (pred[mask] / gt[mask]))
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25**2).float().mean()
        a3 = (thresh < 1.25**3).float().mean()
        rmse = (gt[mask] - pred[mask]) ** 2
        rmse = torch.sqrt(rmse.mean())

        rmse_log = (torch.log(gt[mask]) - torch.log(pred[mask])) ** 2
        # rmse_log[rmse_log == float("inf")] = float("nan")
        rmse_log = torch.sqrt(rmse_log).nanmean()

        abs_rel = torch.abs(gt - pred)[mask] / gt[mask]
        abs_rel = abs_rel.mean()
        sq_rel = (gt - pred)[mask] ** 2 / gt[mask]
        sq_rel = sq_rel.mean()

        return (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)