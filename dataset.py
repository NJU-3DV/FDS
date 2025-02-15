import random
import os
import json
import cv2
import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import dataclass, field

from pointrix.dataset.colmap_data import ColmapDataset
from pointrix.dataset.base_data import DATA_SET_REGISTRY, BaseDataset
from pointrix.logger.writer import Logger, ProgressLogger
from pointrix.dataset.utils.dataprior import CameraPrior, PointsPrior
from pointrix.dataset.utils.dataset import load_from_json
from pointrix.dataset.utils.colmap  import (
    read_colmap_extrinsics,
    read_colmap_intrinsics,
    ExtractColmapCamInfo,
    read_3D_points_binary
)

@DATA_SET_REGISTRY.register()
class SparseDTUDataset(ColmapDataset):
    
    @dataclass
    class Config(ColmapDataset.Config):
        """
        Parameters
        ----------
        data_path: str
            The path to the data
        data_set: str
            The dataset used in the pipeline, indexed in DATA_SET_REGISTRY
        observed_data_dirs_dict: Dict[str, str]
            The observed data directories, e.g., {"image": "images"}, which means the variable image is stored in "images" directory
        cached_observed_data: bool
            Whether the observed data is cached
        white_bg: bool
            Whether the background is white
        enable_camera_training: bool
            Whether the camera is trainable
        scale: float
            The image scale of the dataset
        device: str
            The device used in the pipeline
        """
        splithold: int = 8
        depth_scale: float = 1000.
        bg: float = 1.0
        
    def load_camera_prior(self, split: str) -> List[CameraPrior]:
        """
        The function for loading the camera information.
        
        Parameters:
        -----------
        split: str
            The split of the dataset.

        """
        extrinsics = read_colmap_extrinsics(self.data_root / Path("sparse/0") / Path("images.bin"))
        intrinsics = read_colmap_intrinsics(self.data_root / Path("sparse/0") / Path("cameras.bin"))
        # TODO: more methods for splitting the data
        cameras = []
        for idx, key in enumerate(extrinsics):
            colmapextr = extrinsics[key]
            colmapintr = intrinsics[colmapextr.camera_id]
            R, T, fx, fy, cx, cy, width, height = ExtractColmapCamInfo(colmapextr, colmapintr, self.scale)
            rgb_file_path = os.path.join(self.data_root, "images", os.path.basename(colmapextr.name))
            camera = CameraPrior(idx=idx, R=R, T=T, image_width=width, image_height=height, rgb_file_name=os.path.basename(colmapextr.name), rgb_file_path=rgb_file_path,
                            fx=fx, fy=fy, cx=cx, cy=cy, device='cuda')
            cameras.append(camera)
        sorted_camera = sorted(cameras.copy(), key=lambda x: x.rgb_file_name)
        index = list(range(len(sorted_camera)))
        self.train_index = [25, 22, 28, 40, 44, 48, 0, 8, 13]
        exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
        self.val_index = [i for i in np.arange(49) if i not in self.train_index + exclude_idx]
        self.train_index = self.train_index[:3]
        self.val_index = self.train_index
        cameras_results = [sorted_camera[i] for i in self.train_index] if split == 'train' else [sorted_camera[i] for i in self.val_index] 
        return cameras_results
        
    def load_data_list(self, split: str):
        """
        The foundational function for formating the data

        Parameters
        ----------
        split: The split of the data.
        
        Returns
        -------
        camera: List[CameraPrior]
            The list of cameras prior
        observed_data: Dict[str, Any]
            The observed data
        pointcloud: PointsPrior
            The pointcloud for the gaussian model.
        """
        self.cameras = self.load_camera_prior(split=split)
        self.observed_data = self.load_observed_data(split=split)
        self.pointcloud = self.load_pointcloud_prior()
        return self.cameras, self.observed_data, self.pointcloud
    
    def load_pointcloud_prior(self) -> dict:
        """
        The function for loading the Pointcloud for initialization of gaussian model.

        Returns:
        --------
        point_cloud : dict
            The point cloud for the gaussian model.
        """
        points3d_ply_path = self.data_root / Path("vis/mvs_input.ply")
        point_cloud = PointsPrior()
        point_cloud.read_ply(points3d_ply_path)
        point_cloud.colors = point_cloud.colors / 255.
        return point_cloud
    
    def transform_observed_data(self, observed_data, split):
        cached_progress = ProgressLogger(description='transforming cached observed_data', suffix='iters/s')
        cached_progress.add_task(f'Transforming', f'Transforming {split} cached observed_data', len(observed_data))
        mean_depth = 0
        
        with cached_progress.progress as progress:
            for i in range(len(observed_data)):
                # Transform Image
                image = observed_data[i]['image']
                w, h = image.size
                ori_w, ori_h = w, h
                image = image.resize((int(w * self.scale), int(h * self.scale)))
                image = np.array(image) / 255.
                if image.shape[2] == 4:
                    mask = image[:, :, 3]
                    bg = 1.0 if self.cfg.white_bg else 0.0
                    image = image[:, :, :3] * image[:, :, 3:4] + bg * (1 - image[:, :, 3:4])
                    observed_data[i]['mask'] = torch.from_numpy(np.array(mask)).float().clamp(0.0, 1.0)
                observed_data[i]['image'] = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().clamp(0.0, 1.0)
                
                # Transform Normal
                if "depth_mvs" in observed_data[i]:
                    depth_mvs = observed_data[i]['depth_mvs']
                    h, w = depth_mvs.shape
                    observed_data[i]['depth_mvs'] = (torch.from_numpy(np.array(depth_mvs))).float()
                    observed_data[i]['depth_mvs'] = torch.nn.functional.interpolate(observed_data[i]['depth_mvs'].unsqueeze(0).unsqueeze(0), (int(ori_h * self.scale), int(ori_w * self.scale))).squeeze()
                
                if "mvs_mask" in observed_data[i]:
                    mvs_mask = observed_data[i]['mvs_mask']
                    h, w = mvs_mask.shape
                    observed_data[i]['mvs_mask'] = (torch.from_numpy(np.array(mvs_mask))).float()
                    observed_data[i]['mvs_mask'] = torch.nn.functional.interpolate(observed_data[i]['mvs_mask'].unsqueeze(0).unsqueeze(0).float(), (int(ori_h * self.scale), int(ori_w * self.scale)), mode='nearest').squeeze()

                cached_progress.update(f'Transforming', step=1)
        return observed_data
    
    def __getitem__(self, idx):
        camera = self.camera_list[idx]
        observed_data = self.observed_data[idx]
        frame_idx = self.frame_idx_list[idx]
        return {
            **observed_data,
            "camera": camera,
            "frame_idx": frame_idx,
            "camera_idx": int(camera.idx),
            "height": int(camera.image_height),
            "width": int(camera.image_width)
        }
    
    def load_observed_data(self, split):
        """
        The function for loading the observed_data.

        Parameters:
        -----------
        split: str
            The split of the dataset.
        
        Returns:
        --------
        observed_data: List[Dict[str, Any]]
            The observed_datafor the dataset.
        """
        observed_data = []
        for k, v in self.observed_data_dirs_dict.items():
            cached_progress = ProgressLogger(description='Loading cached observed_data', suffix='iters/s')
            cached_progress.add_task(f'cache_{k}', f'Loading {split} cached {k}', len(self.cameras))
            with cached_progress.progress as progress:
                for i, camera in enumerate(self.cameras):
                    if len(observed_data) <= i:
                        observed_data.append({})
                    if k == 'image':
                        image = np.array(Image.open(camera.rgb_file_path))
                        image = Image.fromarray((image).astype(np.uint8))
                        observed_data[i].update({k: image})
                    if k == 'depth_mvs':
                        depth_mvs_path = str(camera.rgb_file_path).replace("images", "mvs").replace(".png", ".npy")
                        observed_data[i].update({k: np.load(depth_mvs_path)})
                    if k == 'mvs_mask':
                        matches_path = str(camera.rgb_file_path).replace("images", "masks").replace(".png", "_mask.npy")
                        observed_data[i].update({k: np.load(matches_path)})
                    cached_progress.update(f'cache_{k}', step=1)
                    
        return observed_data


@DATA_SET_REGISTRY.register()
class MushRoomDataset(ColmapDataset):
    @dataclass
    class Config(ColmapDataset.Config):
        """
        Parameters
        ----------
        data_path: str
            The path to the data
        data_set: str
            The dataset used in the pipeline, indexed in DATA_SET_REGISTRY
        observed_data_dirs_dict: Dict[str, str]
            The observed data directories, e.g., {"image": "images"}, which means the variable image is stored in "images" directory
        cached_observed_data: bool
            Whether the observed data is cached
        white_bg: bool
            Whether the background is white
        enable_camera_training: bool
            Whether the camera is trainable
        scale: float
            The image scale of the dataset
        device: str
            The device used in the pipeline
        """
        depth_scale: float = 1000.
        bg: float = 1.0
    def load_camera_prior(self, split: str) -> List[CameraPrior]:
        self.MAX_AUTO_RESOLUTION = 1600
        self.orientation_method = "none"
        self.center_method = "none"
        self.downscale_factor = None
        self.iphone_ply_name = Path("iphone_pointcloud.ply")
        self.use_faro_scanner_depths = False
        self.use_faro_scanner_pd = False
        self.data_root = Path(self.data_root)
        
        long_data_dir = self.data_root / "long_capture"
        short_data_dir = self.data_root / "short_capture"
        
        long_meta = load_from_json(long_data_dir / "transformations_colmap.json")
        short_meta = load_from_json(short_data_dir / "transformations_colmap.json")
        
        self.long_meta = long_meta
        self.short_meta = short_meta
        self.long_data_dir = long_data_dir
        self.short_data_dir = short_data_dir
        
        fx_fixed = "fl_x" in long_meta
        fy_fixed = "fl_y" in long_meta
        cx_fixed = "cx" in long_meta
        cy_fixed = "cy" in long_meta
        height_fixed = "h" in long_meta
        width_fixed = "w" in long_meta
        distort_fixed = False
        
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in long_meta:
                distort_fixed = True
                break
            
        long_fnames = []
        for frame in long_meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, long_data_dir)
            long_fnames.append(fname)
        

        inds = np.argsort(long_fnames)
        
            
        long_frames = [long_meta["frames"][ind] for ind in inds]
        
        short_fnames = []
        for frame in short_meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, short_data_dir)
            short_fnames.append(fname)

        inds = np.argsort(short_fnames)

        short_frames = [short_meta["frames"][ind] for ind in inds]
        
        (
            long_filenames,
            long_mask_filenames,
            long_depth_filenames,
            long_poses,
            long_fx,
            long_fy,
            long_cx,
            long_cy,
            long_height,
            long_width,
            long_distort,
        ) = self.get_ele_from_meta(
            long_frames,
            long_data_dir,
            fx_fixed,
            fy_fixed,
            cx_fixed,
            cy_fixed,
            height_fixed,
            width_fixed,
            distort_fixed,
            self.use_faro_scanner_depths,
        )
        (
            short_filenames,
            short_mask_filenames,
            short_depth_filenames,
            short_poses,
            short_fx,
            short_fy,
            short_cx,
            short_cy,
            short_height,
            short_width,
            short_distort,
        ) = self.get_ele_from_meta(
            short_frames,
            short_data_dir,
            fx_fixed,
            fy_fixed,
            cx_fixed,
            cy_fixed,
            height_fixed,
            width_fixed,
            distort_fixed,
            self.use_faro_scanner_depths,
        )
        self.long_filenames = long_filenames
        self.short_filenames = short_filenames
        
        image_filenames = long_filenames + short_filenames
        mask_filenames = long_mask_filenames + short_mask_filenames
        depth_filenames = long_depth_filenames + short_depth_filenames
        poses = long_poses + short_poses
        
        
        fx = long_fx + short_fx
        fy = long_fy + short_fy
        cx = long_cx + short_cx
        cy = long_cy + short_cy
        height = long_height + short_height
        width = long_width + short_width
        distort = long_distort + short_distort
        
        poses = np.array(poses)
        
        # Mushroom eval images
        eval_image_txt_path = Path(long_data_dir / "test.txt")
        test_filenames = []
        
        if eval_image_txt_path.exists():
            with open(eval_image_txt_path) as fid:
                while True:
                    img_name = fid.readline()
                    if not img_name:
                        break
                    img_name = img_name.strip()

                    file_name = "images/" + img_name + ".jpg"


                    test_filenames.append(
                        self._get_fname(
                            file_name,
                            data_dir=long_data_dir,
                            downsample_folder_prefix="images",
                        )
                    )
        else:
            print(
                f"[yellow]Path to test images at {eval_image_txt_path} does not exist. Using zero test images."
            )
            
        i_train, i_eval = self.mushroom_get_train_eval_split_filename(
            long_filenames, test_filenames
        )
        
        if split == "train":
            indices = i_train
        else:
            indices = i_eval
            
        
        if "orientation_override" in long_meta:
            orientation_method = long_meta["orientation_override"]
            print(
                f"[yellow] Dataset is overriding orientation method to {orientation_method}"
            )
        else:
            orientation_method = "none"
            
        poses = torch.from_numpy(poses.astype(np.float32))
        image_filenames = [image_filenames[i] for i in indices]
        self.depth_filenames = (
            [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        )
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]
        
        fx = (
            float(long_meta["fl_x"])
            if fx_fixed
            else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        )
        fy = (
            float(long_meta["fl_y"])
            if fy_fixed
            else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        )
        cx = (
            float(long_meta["cx"])
            if cx_fixed
            else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        )
        cy = (
            float(long_meta["cy"])
            if cy_fixed
            else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        )
        height = (
            int(long_meta["h"])
            if height_fixed
            else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        )
        width = (
            int(long_meta["w"])
            if width_fixed
            else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        )
        self.cameras = []
        OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        for i in range(len(image_filenames)):

            pose = np.matmul(np.array(poses[i]), OPENGL_TO_OPENCV)
            R = pose[:3, :3]
            T = pose[:3, 3]
            R = R.T
            T = -R @ T
            camera = CameraPrior(
                idx=i,
                R=R,
                T=T,
                image_width=width if width_fixed else width[i],
                image_height=height if height_fixed else height[i],
                rgb_file_name=image_filenames[i],
                fx=fx if fx_fixed else fx[i],
                fy=fy if fy_fixed else fy[i],
                cx=cx if cx_fixed else cx[i],
                cy=cy if cy_fixed else cy[i],
                device="cuda",
            )
            self.cameras.append(camera)
        return self.cameras
        
    
    def _get_fname(
        self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_"
    ) -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if self.downscale_factor is None:
            test_img = Image.open(data_dir / filepath)
            h, w = test_img.size
            max_res = max(h, w)
            df = 0
            while True:
                if (max_res / 2 ** (df)) < self.MAX_AUTO_RESOLUTION:
                    break
                if not (
                    data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name
                ).exists():
                    break
                df += 1

            self.downscale_factor = 2**df
            # TODO check if there is a better way to inform user of downscale factor instead of printing so many lines
            # CONSOLE.print(f"Auto image downscale factor of {self.downscale_factor}")
        else:
            self.downscale_factor = self.downscale_factor

        if self.downscale_factor > 1:
            return (
                data_dir
                / f"{downsample_folder_prefix}{self.downscale_factor}"
                / filepath.name
            )
        return data_dir / filepath
    
    def get_ele_from_meta(
        self,
        frames,
        data_dir,
        fx_fixed,
        fy_fixed,
        cx_fixed,
        cy_fixed,
        height_fixed,
        width_fixed,
        distort_fixed,
        use_faro_scanner_depths,
    ):
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath, data_dir, downsample_folder_prefix="masks_"
                )
                mask_filenames.append(mask_fname)

            if "depth_file_path" in frame:
                if use_faro_scanner_depths:
                    depth_filepath = Path(
                        frame["depth_file_path"].replace("depths", "reference_depth")
                    )
                else:
                    depth_filepath = Path(frame["depth_file_path"])
                depth_fname = self._get_fname(
                    depth_filepath, data_dir, downsample_folder_prefix="depths_"
                )
                depth_filenames.append(depth_fname)

        return (
            image_filenames,
            mask_filenames,
            depth_filenames,
            poses,
            fx,
            fy,
            cx,
            cy,
            height,
            width,
            distort,
        )
        
    def mushroom_get_train_eval_split_filename(
        self, image_filenames: List, test_filenames: List
    ):
        """
        Get the train/eval split based on the filename of the images.

        Args:
            image_filenames: list of image filenames
        """
        if not test_filenames:
            num_images = len(image_filenames)
            return np.arange(num_images), np.arange(0)
        num_images = len(image_filenames)
        basenames = [
            os.path.basename(image_filename) for image_filename in image_filenames
        ]
        test_basenames = [
            os.path.basename(test_filename) for test_filename in test_filenames
        ]
        i_all = np.arange(num_images)
        i_train = []
        i_eval = []
        for idx, basename in zip(i_all, basenames):
            # check the frame index
            if basename in test_basenames:
                i_eval.append(idx)
            else:
                i_train.append(idx)

        return np.array(i_train), np.array(i_eval)
    
    def load_pointcloud_prior(self) -> dict:
        """
        The function for loading the Pointcloud for initialization of gaussian model.

        Returns:
        --------
        point_cloud : dict
            The point cloud for the gaussian model.
        """
        points3d_bin_path = self.long_data_dir / Path("points3D.bin")
        point_cloud = PointsPrior()
        positions, colors = read_3D_points_binary(points3d_bin_path)
        normals = np.zeros_like(positions)
        point_cloud = PointsPrior(positions=positions, colors=colors/255., normals=normals)
        return point_cloud
    
    def load_observed_data(self, split):
        """
        The function for loading the observed_data.

        Parameters:
        -----------
        split: str
            The split of the dataset.
        
        Returns:
        --------
        observed_data: List[Dict[str, Any]]
            The observed_datafor the dataset.
        """
        observed_data = []
        for k, v in self.observed_data_dirs_dict.items():
            cached_progress = ProgressLogger(description='Loading cached observed_data', suffix='iters/s')
            cached_progress.add_task(f'cache_{k}', f'Loading {split} cached {k}', len(self.cameras))
            with cached_progress.progress as progress:
                for i, camera in enumerate(self.cameras):
                    if len(observed_data) <= i:
                        observed_data.append({})
                    if k == 'image':
                        image = np.array(Image.open(camera.rgb_file_name))
                        image = Image.fromarray((image).astype(np.uint8))
                        observed_data[i].update({k: image})
                    if k == 'depth':
                        depth_path = str(camera.rgb_file_name).replace("images", "depth").replace(".jpg", ".png")
                        observed_data[i].update({k: Image.open(depth_path)})
                    if k == 'normal':
                        normal_path = str(camera.rgb_file_name).replace("images", v).replace(".jpg", ".png")
                        observed_data[i].update({k: Image.open(normal_path)})
                    if k == 'depth_mono':
                        depth_mono_path = str(camera.rgb_file_name).replace("images", "depth_mono").replace(".jpg", ".png")
                        observed_data[i].update({k: Image.open(depth_mono_path)})
                    if k == 'depth_mvs':
                        depth_mvs_path = str(camera.rgb_file_name).replace("images", "depth_mvs").replace(".jpg", ".png")
                        observed_data[i].update({k: Image.open(depth_mvs_path)})
                    if k == 'matches':
                        matches_path = str(camera.rgb_file_name).replace("images", "matches").replace(".jpg", ".npy")
                        observed_data[i].update({k: np.load(matches_path)})
                    cached_progress.update(f'cache_{k}', step=1)
                    
        return observed_data

    def transform_observed_data(self, observed_data, split):
        cached_progress = ProgressLogger(description='transforming cached observed_data', suffix='iters/s')
        cached_progress.add_task(f'Transforming', f'Transforming {split} cached observed_data', len(observed_data))
        mean_depth = 0
        
        with cached_progress.progress as progress:
            for i in range(len(observed_data)):
                # Transform Image
                image = observed_data[i]['image']
                w, h = image.size
                image = image.resize((int(w * self.scale), int(h * self.scale)))
                image = np.array(image) / 255.
                if image.shape[2] == 4:
                    mask = image[:, :, 3]
                    bg = 1.0 if self.cfg.white_bg else 0.0
                    image = image[:, :, :3] * image[:, :, 3:4] + bg * (1 - image[:, :, 3:4])
                    observed_data[i]['mask'] = torch.from_numpy(np.array(mask)).float().clamp(0.0, 1.0)
                observed_data[i]['image'] = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().clamp(0.0, 1.0)
                cached_progress.update(f'Transforming', step=1)
                if "depth" in observed_data[i]:
                    depth = observed_data[i]['depth']
                    w, h = depth.size
                    depth = depth.resize((int(w * self.scale), int(h * self.scale)))
                    observed_data[i]['depth'] = (torch.from_numpy(np.array(depth, dtype=np.int32)) / self.cfg.depth_scale).float().unsqueeze(0)
                    mean_depth += torch.mean(observed_data[i]['depth'])
        return observed_data