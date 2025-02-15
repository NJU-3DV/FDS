import cv2
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
from typing import Dict, Literal, Optional, Union

from kornia.geometry.depth import depth_to_3d_v2
from kornia.utils import create_meshgrid
from metric import DepthMetrics
from pointrix.model.base_model import BaseModel, MODEL_REGISTRY
from pointrix.model.camera.camera_model import CameraModel, CAMERA_REGISTRY
from pointrix.model.loss import l1_loss, ssim, psnr, l2_loss
from pointrix.utils.pose import Fov2ProjectMat

from SEA_RAFT.core.raft import RAFT
from SEA_RAFT.core.utils.utils import InputPadder


@CAMERA_REGISTRY.register()
class FDS_camera(CameraModel):
    @dataclass
    class Config(CameraModel.Config):
        radius: float=10.0
    cfg: Config
        
    def fds_sample(self, idx_list, mean_disp, cnt=1):
        render_poses = []
        E_cuda = self.extrinsic_matrices(idx_list)
        E_s = E_cuda.detach().cpu().numpy()
            
        fc = self.intrinsic_params(idx_list)[0, 0].detach().cpu().numpy()
        
        radius = self.cfg.radius

        max_trans = (radius - 0.0) * mean_disp / fc

        for i in range(cnt):
            # Assume len(idx_list) == 1
            # [C, 4, 4]
            i = np.random.uniform(0, 60)
            
            x_trans = max_trans * 3.0 * np.sin(2.0 * np.pi * float(i) / float(60))
            y_trans = max_trans * 3.0 * np.cos(2.0 * np.pi * float(i) / float(60))
            z_trans = 0.0
    
            i_pose = np.concatenate(
                [
                    np.concatenate(
                        [np.eye(3), np.array([x_trans, y_trans, z_trans])[
                            :, np.newaxis]],
                        axis=1,
                    ),
                    np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
                ],
                axis=0,
            )
            render_pose = np.linalg.inv(np.dot(np.linalg.inv(E_s), i_pose))
            render_poses.append(render_pose)
        render_poses = np.concatenate(render_poses, axis=0)
        return torch.Tensor(render_poses).to(E_cuda.device)

@MODEL_REGISTRY.register()
class FDSGaussianModel(BaseModel):
    @dataclass
    class Config(BaseModel.Config):
        enable_flow_loss: bool = True
        flow_loss_weight: float = 0.02
        flow_start_iter: int = 15000
        normal_loss_weight: float = 0.25
        occlusion_threshold: float = 0.08
        render_normal_loss_weight: float = 0.2
        raft_path: str = "/NASdata/clz/pretrained_weights/Tartan-C-T-TSKH432x960-M.pth"
        raft_downsample_ratio: float =  0.5
        dist_loss_weight: float = 0.0
    cfg: Config

    def setup(self, datapipeline, device="cuda"):
        super().setup(datapipeline, device)
        
        raft_offline_config = {"use_var": True, "var_min": 0, 
                                "var_max": 10, "pretrain": "resnet34", 
                                "initial_dim": 64, "block_dims": [64, 128, 256], 
                                "radius": 4, "dim": 128, "num_blocks": 2, "iters": 12}
        args = argparse.Namespace(**raft_offline_config)
        self.flow_model = RAFT(args)
        self.flow_model.load_state_dict(torch.load(self.cfg.raft_path))
        self.flow_model.to(self.device)
        self.flow_model.eval()

        self.depth_metrics = DepthMetrics()
    
    def forward(self, batch=None, training=True, render=True, iteration=None, render_fds=False) -> dict:
        if iteration is not None:
            self.renderer.update_sh_degree(iteration)

        if batch is None:
            return {
                    "position": self.point_cloud.position,
                    "opacity": self.point_cloud.get_opacity,
                    "scaling": self.point_cloud.get_scaling,
                    "rotation": self.point_cloud.get_rotation,
                    "shs": self.point_cloud.get_shs,
                }
    
        frame_idx_list = [batch[i]["frame_idx"] for i in range(len(batch))]
        extrinsic_matrix = self.training_camera_model.extrinsic_matrices(frame_idx_list) \
            if training else self.validation_camera_model.extrinsic_matrices(frame_idx_list)
        intrinsic_params = self.training_camera_model.intrinsic_params(frame_idx_list) \
            if training else self.validation_camera_model.intrinsic_params(frame_idx_list)
        camera_center = self.training_camera_model.camera_centers(frame_idx_list) \
            if training else self.validation_camera_model.camera_centers(frame_idx_list)
        
        
        render_dict = {
            "extrinsic_matrix": extrinsic_matrix,
            "intrinsic_params": intrinsic_params,
            "camera_center": camera_center,
            "position": self.point_cloud.position,
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling,
            "rotation": self.point_cloud.get_rotation,
            "shs": self.point_cloud.get_shs,
            "height": batch[0]['height'],
            "width": batch[0]['width']
        }

        if render:
            fds_render_results = None
            render_results = self.renderer.render_batch(render_dict)

            if self.cfg.enable_flow_loss and iteration is not None and iteration >= self.cfg.flow_start_iter:
                if "mask" in batch[0]:
                    masks = torch.stack(
                        [batch[i]["mask"] for i in range(len(batch))],
                        dim=0
                    )
                    masks = masks.unsqueeze(1)
                    depth_mean = (torch.sum(render_results['depth'] * masks) / masks.sum()).detach().cpu().numpy()
                else:
                    depth_mean = torch.mean(render_results['depth']).detach().cpu().numpy()

                
                fds_extrinsic_matrix = self.training_camera_model.fds_sample(frame_idx_list, depth_mean) \
                    if training else self.validation_camera_model.fds_sample(frame_idx_list, depth_mean)
                
                fds_extrinsic_matrix = fds_extrinsic_matrix.detach()
                fds_camera_center = fds_extrinsic_matrix.inverse()[:, :3, 3]
                fds_render_dict = {
                    "extrinsic_matrix": fds_extrinsic_matrix,
                    "intrinsic_params": intrinsic_params,
                    "camera_center": fds_camera_center,
                    "position": self.point_cloud.position,
                    "opacity": self.point_cloud.get_opacity,
                    "scaling": self.point_cloud.get_scaling,
                    "rotation": self.point_cloud.get_rotation,
                    "shs": self.point_cloud.get_shs,
                    "height": batch[0]['height'],
                    "width": batch[0]['width']
                }

                fds_render_results = self.renderer.render_batch(fds_render_dict)

                gt_flow_depth, mask= calculate_flow_depth(render_results['depth'],
                                                            fds_render_results['depth'],
                                                            extrinsic_matrix[0], fds_extrinsic_matrix, 
                                                            intrinsic_params[0], int(self.training_camera_model.image_height), 
                                                            int(self.training_camera_model.image_width), self.cfg.occlusion_threshold) # [2, H, W]

                render_results['flow'] = gt_flow_depth
                render_results['mask'] = mask

                with torch.no_grad():
                    if iteration is not None and iteration >= self.cfg.flow_start_iter and self.cfg.enable_flow_loss:
                        image_flow, image_flow_neighbor = batch[0]['image'].unsqueeze(0)*255, fds_render_results['rgb']*255.
                        image_flow = image_flow.repeat(fds_render_results['rgb'].shape[0], 1, 1, 1)

                        if self.cfg.raft_downsample_ratio == 1.0:
                            image_flow_downsampled = image_flow
                            image_flow_neighbor_downsampled = image_flow_neighbor
                        else:
                            image_flow_downsampled = F.interpolate(image_flow, scale_factor=self.cfg.raft_downsample_ratio, mode='bilinear', align_corners=False)
                            image_flow_neighbor_downsampled = F.interpolate(image_flow_neighbor, scale_factor=self.cfg.raft_downsample_ratio, mode='bilinear', align_corners=False)

                        padder = InputPadder(image_flow_neighbor_downsampled.shape)
                        image_flow_downsampled, image_flow_neighbor_downsampled = padder.pad(image_flow_downsampled, image_flow_neighbor_downsampled)

                        output = self.flow_model(image1=image_flow_downsampled, image2=image_flow_neighbor_downsampled, iters=12, test_mode=True)

                        flow_raft = output['flow'][-1]
                        flow_raft = padder.unpad(flow_raft)

                        if self.cfg.raft_downsample_ratio != 1.0:
                            flow_raft = flow_raft * (1.0/self.cfg.raft_downsample_ratio)
                            flow_raft = F.interpolate(flow_raft, scale_factor=1.0/self.cfg.raft_downsample_ratio, mode='bilinear', align_corners=False)
                        render_results['flow_raft'] = flow_raft
            render_results['fds_render_results'] = fds_render_results
            return render_results
        else:
            return render_dict

    def get_loss_dict(self, results, batch, step) -> dict:
        render_results = results
        loss_dict = {}
        loss = 0.0
        
        gt_images = torch.stack(
            [batch[i]["image"] for i in range(len(batch))],
            dim=0
        )
        if "normal" in batch[0]:
            normal_images = torch.stack(
                [batch[i]["normal"] for i in range(len(batch))],
                dim=0
            )
        if "mask" in batch[0]:
            # [1, H, W]
            masks = torch.stack(
                [batch[i]["mask"] for i in range(len(batch))],
                dim=0
            )
            masks = masks.unsqueeze(1)
            render_results['mask_color'] = masks.squeeze()
            if "mask" in render_results:
                render_results['mask'] = render_results['mask'] * masks
        
        L1_loss = l1_loss(render_results['rgb'], gt_images)
        ssim_loss = 1.0 - ssim(render_results['rgb'], gt_images)
        loss += (1.0 - self.cfg.lambda_ssim) * L1_loss
        loss += self.cfg.lambda_ssim * ssim_loss
        loss_dict.update({"L1_loss": L1_loss})
        loss_dict.update({"ssim_loss": ssim_loss})

        if step > self.cfg.flow_start_iter:
            # normal_images = normal_images * 2 - 1.
            normal_error = (1 - (render_results['normal'] * render_results['depth_normal']).sum(dim=1))[None]
            normal_loss = self.cfg.normal_loss_weight * (normal_error).mean()
            loss += normal_loss
            loss_dict.update({"normal_loss": normal_loss})
        
        if step > self.cfg.flow_start_iter and self.cfg.enable_flow_loss:
            flow_loss = l1_loss(render_results['flow_raft'].squeeze().detach(
            ), render_results['flow'].squeeze(), return_mean=False) * self.cfg.flow_loss_weight * render_results['mask'].squeeze()
            flow_loss = (flow_loss[~torch.isnan(flow_loss)]).sum() / render_results['mask'].sum()
            loss += flow_loss
            loss_dict.update({"flow_loss": flow_loss})
        
        if "render_dist" in render_results and step >= 3000:
            dist_loss = self.cfg.dist_loss_weight * (render_results["render_dist"]).mean()
            loss += dist_loss
            loss_dict.update({"dist_loss": dist_loss})

        loss_dict.update({"loss": loss})
        return loss_dict

    @torch.no_grad()
    def get_metric_dict(self, render_results, batch) -> dict:
        gt_images = torch.clamp(torch.stack(
            [batch[i]["image"].to(self.device) for i in range(len(batch))],
            dim=0), 0.0, 1.0)
        
        gt_depths = None
        if "depth" in batch[0]:
            gt_depths = torch.stack(
                [batch[i]["depth"] for i in range(len(batch))],
                dim=0
            )
        
        if gt_depths is not None:
            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = self.depth_metrics(
                render_results['depth'].squeeze(), gt_depths.squeeze())
        else:
            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = 0., 0., 0., 0., 0., 0., 0.
        
        rgb = torch.clamp(render_results['rgb'], 0.0, 1.0)
        L1_loss = l1_loss(rgb, gt_images).mean().double()
        psnr_test = psnr(rgb.squeeze(), gt_images.squeeze()).mean().double()
        ssims_test = ssim(rgb, gt_images, size_average=True).mean().item()
        lpips_vgg_test = self.lpips_func(rgb, gt_images).mean().item()
        metric_dict = {"L1_loss": L1_loss,
                       "psnr": psnr_test,
                       "ssims": ssims_test,
                       "lpips": lpips_vgg_test,
                       "gt_images": gt_images,
                       "images": rgb,
                       "rgb_file_name": batch[0]["camera"].rgb_file_name,
                       "abs_rel": abs_rel,
                       "sq_rel": sq_rel,
                       "rmse": rmse,
                       "rmse_log": rmse_log,
                       "a1": a1,
                       "a2": a2,
                       "a3": a3}

        if 'depth' in render_results:
            depth = render_results['depth']
            metric_dict['depth'] = depth
            # if 'mask_color' in render_results:
            #     metric_dict['depth'] = metric_dict['depth'] * render_results['mask_color'] + -1.0 * (1 - render_results['mask_color'])

        if 'normal_im' in render_results:
            normal_im = render_results['normal_im']
            metric_dict['normal_im'] = normal_im
            
            normal = render_results['normal']
            metric_dict['normal'] = normal
            
            depth_normal = render_results['depth_normal']
            metric_dict['depth_normal'] = depth_normal
            
        if "mask" in render_results:
            mask = render_results['mask'].squeeze(0)[0, ...]
            metric_dict['mask'] = mask
            
        if 'normal' in batch[0]:
            normal = batch[0]['normal']
            metric_dict['normal_gt'] = normal
            
        if 'flow' in render_results:
            flow = render_results['flow']
            metric_dict['flow'] = flow.squeeze()

        if 'flow_raft' in render_results:
            flow_raft = render_results['flow_raft']
            metric_dict['flow_raft'] = flow_raft[0]
            
        if 'next_flow_raft' in render_results:
            next_flow_raft = render_results['next_flow_raft']
            metric_dict['next_flow_raft'] = next_flow_raft

        return metric_dict


def calculate_flow_depth(depth, ref_depth, E, E_ns, K, H, W, occlusion_threshold=0.5):
    fx, fy = K[0], K[1]
    fovx = 2*math.atan(W/(2*fx))
    fovy = 2*math.atan(H/(2*fy))
    projection_matrix = Fov2ProjectMat(fovx, fovy).to(E.device).transpose(0,1)
    
    ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W) / 2],
            [0, H / 2, 0, (H) / 2],
            [0, 0, 0, 1]], device=E.device).float().T
    intrinsics_matrix =  (projection_matrix @ ndc2pix)[:3,:3].T

    # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    cx = intrinsics_matrix[0,2]
    cy = intrinsics_matrix[1,2]
    fx = intrinsics_matrix[0,0]
    fy = intrinsics_matrix[1,1]
    
    world_points = depth_to_3d_v2(depth.squeeze(), intrinsics_matrix).permute(2, 0, 1)# B 3 H W
    world_points = torch.cat([world_points, torch.ones(1, H, W, device=world_points.device)], 0) # B 4 H W, in camera view
    flows = []
    masks = []
    for i in range(E_ns.shape[0]):
        E_n = E_ns[i]
        T = E_n @ E.inverse()
        cam_points = torch.matmul(T, world_points.view(4, -1))
        
        computed_depth = cam_points[2, ...].view(H, W)
        
        cam_points_n = (cam_points[:2, ...] / (cam_points[2, ...]))  ## [4, N]

        uv_n_proj = torch.concat([fx * cam_points_n[0, ...] + cx, fy * cam_points_n[1, ...] + cy], 0) ## [2, N]
        uv_n_proj = uv_n_proj.view(2, H, W)
        uv = create_meshgrid(H, W, normalized_coordinates=False, device=depth.device).squeeze().permute(2, 0, 1).float()
        flow = uv_n_proj - uv

        pix_coords = uv_n_proj.permute(1, 2, 0).unsqueeze(0)
        pix_coords[..., 0] = pix_coords[..., 0] / (W - 1)
        pix_coords[..., 1] = pix_coords[..., 1] / (H - 1)
        pix_coords = (pix_coords - 0.5) * 2
        
        projected_depth = F.grid_sample(ref_depth, pix_coords, padding_mode='zeros', align_corners=False).squeeze()
        occlution = (computed_depth - projected_depth) < occlusion_threshold
        # [H, W]
        mask = (uv_n_proj[0] >= 0) & (uv_n_proj[0] < (W-1)) & (uv_n_proj[1] >= 0) & (uv_n_proj[1] < (H-1)).detach() & occlution
        mask = mask.unsqueeze(0).repeat(2, 1, 1)
        flows.append(flow)
        masks.append(mask)
    # [C, 2, H, W]
    flows = torch.stack(flows, 0)
    masks = torch.stack(masks, 0)
    return flows, masks