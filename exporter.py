import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, List

import json
import math
import os
import multiprocessing as mp
import cv2
import numpy as np
import os
import glob
import trimesh
from skimage import measure
import numpy as np
from torch import Tensor
import open3d as o3d
import torch
import torch.nn.functional as F
import tyro
from tqdm import tqdm

import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse

import os
import random
import numpy as np

import torch
import imageio

from pointrix.exporter.base_exporter import MetricExporter, EXPORTER_REGISTRY
from metric import DepthMetrics
from pointrix.model.loss import psnr, ssim, LPIPS, l1_loss
from pointrix.utils.visualize import visualize_depth, visualize_rgb, visualize_normal, visualize_depth_normal
from pointrix.logger.writer import ProgressLogger


@EXPORTER_REGISTRY.register()
class DepthMetricExporter(MetricExporter):
    """
    Base class for all exporters.

    Parameters
    ----------
    cfg : Optional[Union[dict, DictConfig]]
        The configuration dictionary.
    model : BaseModel
        The model which is used to render
    datapipeline : BaseDataPipeline
        The data pipeline which is used to initialize the point cloud.
    device : str, optional
        The device to use, by default "cuda".
    """
    
    def setup(self, model, datapipeline, device="cuda"):
        super().setup(model, datapipeline, device)
        self.reset_metric()
        self.depth_metrics = DepthMetrics()
        
    def reset_metric(self):
        self.l1 = 0.0
        self.psnr_metric = 0.0
        self.ssim_metric  = 0.0
        self.lpips_metric  = 0.0
        
        self.abs_rel_metric = 0.0
        self.sq_rel_metric = 0.0
        self.rmse_metric = 0.0
        self.rmse_log_metric = 0.0
        self.a1_metric = 0.0
        self.a2_metric = 0.0
        self.a3_metric = 0.0
        
    @torch.no_grad()
    def forward(self, output_path):
        """
        Render the test view and save the images to the output path.

        Parameters
        ----------
        model : BaseModel
            The point cloud model.
        datapipeline : DataPipeline
            The data pipeline object.
        output_path : str
            The output path to save the images.
        """
        self.reset_metric()
        lpips_func = LPIPS()
        val_dataset = self.datapipeline.validation_dataset
        val_dataset_size = len(val_dataset)
        progress_logger = ProgressLogger(description='Extracting metrics', suffix='iters/s')
        progress_logger.add_task(f'Metric', f'Extracting metrics', val_dataset_size)
        os.makedirs(os.path.join(output_path, 'test_view'), exist_ok=True)

        with progress_logger.progress as progress:
            for i in range(0, val_dataset_size):
                batch = self.datapipeline.next_val(i)
                render_results = self.model(batch, training=False)
                image_name = os.path.basename(batch[0]['camera'].rgb_file_name)
                gt = torch.clamp(batch[0]['image'].to("cuda").float(), 0.0, 1.0)
                image = torch.clamp(
                    render_results['rgb'], 0.0, 1.0).squeeze()
                visualize_feature = ['rgb', 'depth', 'normal', 'depth_normal']
                depth_gt = batch[0]['depth'].to("cuda").float()
                
                gt_results = {}
                gt_results['rgb'] = gt
                gt_results['depth'] = depth_gt

                for feat_name in visualize_feature:
                    feat = render_results[feat_name]
                    try:
                        visual_feat = eval(f"visualize_{feat_name}")(feat.squeeze())
                    except:
                        visual_feat = feat.squeeze()
                    if not os.path.exists(os.path.join(output_path, f'test_view_{feat_name}')):
                        os.makedirs(os.path.join(
                            output_path, f'test_view_{feat_name}'))
                    imageio.imwrite(os.path.join(
                        output_path, f'test_view_{feat_name}', image_name), visual_feat)
                
                for feat_name, feat in gt_results.items():
                    try:
                        visual_feat = eval(f"visualize_{feat_name}")(feat.squeeze())
                    except:
                        visual_feat = feat.squeeze()
                    if not os.path.exists(os.path.join(output_path, f'gt_{feat_name}')):
                        os.makedirs(os.path.join(
                            output_path, f'gt_{feat_name}'))
                    imageio.imwrite(os.path.join(
                        output_path, f'gt_{feat_name}', image_name), visual_feat)
                

                self.l1 += l1_loss(image, gt, return_mean=True).double()
                self.psnr_metric += psnr(image, gt).mean().double()
                self.ssim_metric += ssim(image, gt).mean().double()
                self.lpips_metric += lpips_func(image, gt).mean().double()
                
                
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = self.depth_metrics(
                        render_results['depth'].squeeze(), depth_gt.squeeze())
                
                self.abs_rel_metric += abs_rel
                self.sq_rel_metric += sq_rel
                self.rmse_metric += rmse
                self.rmse_log_metric += rmse_log
                self.a1_metric += a1
                self.a2_metric += a2
                self.a3_metric += a3
                
                progress_logger.update(f'Metric', step=1)
        self.l1 /= val_dataset_size
        self.psnr_metric /= val_dataset_size
        self.ssim_metric /= val_dataset_size
        self.lpips_metric /= val_dataset_size
        self.abs_rel_metric /= val_dataset_size
        self.sq_rel_metric /= val_dataset_size
        self.rmse_metric /= val_dataset_size
        self.rmse_log_metric /= val_dataset_size
        self.a1_metric /= val_dataset_size
        self.a2_metric /= val_dataset_size
        self.a3_metric /= val_dataset_size
        
        print(
            f"Test results: L1 {self.l1:.5f} PSNR {self.psnr_metric:.5f} SSIM {self.ssim_metric:.5f}, LPIPS (VGG) {self.lpips_metric:.5f}, abs_rel {self.abs_rel_metric:.5f}, sq_rel {self.sq_rel_metric:.5f}, rmse {self.rmse_metric:.5f}, rmse_log {self.rmse_log_metric:.5f}, a1 {self.a1_metric:.5f}, a2 {self.a2_metric:.5f}, a3 {self.a3_metric:.5f}")