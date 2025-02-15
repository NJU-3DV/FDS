import os

from pointrix.hook.log_hook import LogHook
from pointrix.hook.base_hook import HOOK_REGISTRY
from pointrix.utils.visualize import visualize_depth, visualize_flow

@HOOK_REGISTRY.register()
class NormalLogHook(LogHook):
    
    def __init__(self):
        super().__init__()
        self.losses_test = {"L1_loss": 0., "psnr": 0., "ssims": 0., "lpips": 0., 'abs_rel': 0., 'sq_rel': 0., 'rmse': 0., 'a1': 0., 'a2': 0., 'a3': 0.}
                
    def after_val_iter(self, trainner) -> None:
        self.progress_bar.update("validation", step=1)
        for key, value in trainner.metric_dict.items():
            if key in self.losses_test:
                self.losses_test[key] += value

        image_name = os.path.basename(trainner.metric_dict['rgb_file_name'])
        iteration = trainner.global_step
        if 'depth' in trainner.metric_dict:
            visual_depth = visualize_depth(trainner.metric_dict['depth'].squeeze(), tensorboard=True)
            trainner.writer.write_image(
            "test" + f"_view_{image_name}/depth",
            visual_depth, step=iteration)
        trainner.writer.write_image(
            "test" + f"_view_{image_name}/render",
            trainner.metric_dict['images'].squeeze(),
            step=iteration)

        trainner.writer.write_image(
            "test" + f"_view_{image_name}/ground_truth",
            trainner.metric_dict['gt_images'].squeeze(),
            step=iteration)
        
        trainner.writer.write_image(
            "test" + f"_view_{image_name}/normal",
            trainner.metric_dict['normal'].squeeze(),
            step=iteration)
        if "normal_gt" in trainner.metric_dict:
            trainner.writer.write_image(
                "test" + f"_view_{image_name}/normal_gt",
                trainner.metric_dict['normal_gt'].squeeze(),
                step=iteration)
        
        trainner.writer.write_image(
            "test" + f"_view_{image_name}/depth_normal",
            (trainner.metric_dict['depth_normal'].squeeze() + 1) / 2.,
            step=iteration)
        pass