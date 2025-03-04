# Flow Distillation Sampling: Regularizing 3D Gaussians with Pre-trained Matching Priors
<div align="center">
  <a href=https://nju-3dv.github.io/projects/fds/ target="_blank"><img src=https://img.shields.io/badge/Project%20Page-333399.svg?logo=googlehome height=22px></a>
  <a href=https://nju-3dv.github.io/projects/fds/fds.pdf target="_blank"><img src=https://img.shields.io/badge/Paper-b5212f.svg?logo=paperswithcode height=22px></a>
  <a href=https://arxiv.org/abs/2502.07615 target="_blank"><img src=https://img.shields.io/badge/Arxiv-b5212f.svg?logo=arxiv height=22px></a>
</div>



<p align="center">
<span class="author-block">
                <a href="https://linzhuo.xyz">Lin-Zhuo Chen</a><sup>1 *</sup>&nbsp
              </span>
              <span class="author-block">
                Kangjie Liu</a><sup>1 *</sup>&nbsp</span>
              <span class="author-block">
                <a href="https://linyou.github.io/">Youtian
                  Lin</a><sup>1</sup>&nbsp
              </span>
              <span class="author-block">
                Zhihao Li<sup>2</sup>&nbsp</span>
              <span class="author-block">
                <a href="https://siyuzhu-fudan.github.io/">
                  Siyu Zhu</a><sup>3</sup>&nbsp
              </span>
              <span class="author-block">
                <a href="https://cite.nju.edu.cn/People/Faculty/20190621/i5054.html">
                  Xun Cao</a><sup>1</sup>&nbsp
              </span>
              <span class="author-block">
                <a href="https://yoyo000.github.io/">
                  Yao Yao</a><sup>1 †</sup>&nbsp
              </span>
</p>

This is official implement of our ICLR 2025 paper: **Flow Distillation Sampling: Regularizing 3D Gaussians with Pre-trained Matching Priors**.

## Update
- 2025-03-04: Change the default mesh extraction method.

## 📝 Abstract

3D Gaussian Splatting (3DGS) has achieved excellent rendering quality with fast training and rendering speed. However, its optimization process lacks explicit geometric constraints, leading to suboptimal geometric reconstruction in regions with sparse or no observational input views. In this work, we try to mitigate the issue by incorporating a pre-trained matching prior to the 3DGS optimization process. We introduce Flow Distillation Sampling (FDS), a technique that leverages pre-trained geometric knowledge to bolster the accuracy of the Gaussian radiance field. Our method employs a strategic sampling technique to target unobserved views adjacent to the input views, utilizing the optical flow calculated from the matching model (Prior Flow) to guide the flow analytically calculated from the 3DGS geometry (Radiance Flow). Comprehensive experiments in depth rendering, mesh reconstruction, and novel view synthesis showcase the significant advantages of FDS over state-of-the-art methods. Additionally, our interpretive experiments and analysis aim to shed light on the effects of FDS on geometric accuracy and rendering quality, potentially providing readers with insights into its performance.

## 🚀 Getting Started

### Data preparation
1. Download our colmap points for 2DGS initilization: [mushroom_colmap](https://drive.google.com/drive/folders/1ExkHpQ4wkCDPMXvAn5uuiApvSUII1gVi?usp=drive_link).
2. Download mushroom dataset: [mushroom_website](https://github.com/TUTvision/MuSHRoom).
3. Put our colmap points into mushroom dataset:

```
FDS
├── Mushroom
    ├── activity
    |   ├── iphone
    |   ├── ├── long_capture
    |   ├── ├── ├── put colmap points here

    ├── classroom
    |   ├── iphone
    |   ├── ├── long_capture
    |   ├── ├── ├── put colmap points here

...
```

### Installation

#### Clone FDS
```bash
git clone https://github.com/NJU-3DV/FDS.git --recursive
pip install -r requirements.txt
```

#### Install Pointrix
```
cd pointrix
git submodule update --init --recursive
```
Please refer to https://github.com/pointrix-project/Pointrix for the install instruction.

#### Install renderer kernel of 2dgs
```
git clone https://github.com/hbb1/diff-surfel-rasterization.git
pip install diff-surfel-rasterization
```

#### Pretrain optical flow weight

Download pretrained optical flow weight: Tartan-C-T-TSKH432x960-M.pth in https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW,
and put it into your folder.

### Running

#### Mushroom dataset

```bash
python launch.py --config configs/mushroom_config.yaml \
                  trainer.datapipeline.dataset.data_path=[your_data_path] \
                  trainer.output_path=[your_log_path] \
                  trainer.exporter.exporter_b.extra_cfg.gt_mesh_path=[your_mesh_path]  \
                  trainer.model.raft_path=[your_raft_weight_path]
                  trainer.gui.viewer_port=8005
```

for example, to run vr room scene in mushroom dataset:

```bash
python launch.py --config configs/mushroom_config.yaml \
                        trainer.datapipeline.dataset.data_path=/NASdata/clz/data/mushroom/vr_room/iphone \
                        trainer.output_path=/NASdata/clz/log/fds_paper_final_v2/2dgs/fds_test/vr_room \
                        trainer.exporter.exporter_b.extra_cfg.gt_mesh_path=/NASdata/clz/data/mushroom/vr_room \
                        trainer.gui.viewer_port=8005
```

## TODO
- [ ] More stable results
- [ ] DTU datasets.
- [ ] Supervised with more prior information.

## Acknowledgements

Thanks to the following repos for their great work, which helps us a lot in the development of FDS:

- [Pointrix](https://github.com/Pointrix-Project/Pointrix): A light weight framework for gaussian points rendering.
- [DN Splatter](https://github.com/maturk/dn-splatter): The mesh exporter and mushroom dataloader.
- [2d gaussian splatting](https://github.com/hbb1/2d-gaussian-splatting): Rendering Kernel.
- [Sea Raft](https://github.com/princeton-vl/SEA-RAFT): Optical flow model of FDS.


## Citation

If you find this work is useful for your research, please cite our paper:
```
@inproceedings{chen2024fds, 
        title={Flow Distillation Sampling: Regularizing 3D Gaussians with Pre-trained Matching Priors}, 
        author={Lin-Zhuo Chen and Kangjie Liu and Youtian Lin and Zhihao Li and Siyu Zhu and Xun Cao and Yao Yao}, 
        booktitle={ICLR}, 
        year={2025}
      }
```
