import fastmri
import torch
import numpy as np
import os
from utils.cal_ssim import ssim as py_msssim
import matplotlib.pyplot as plt
# from segment.seg_test import get_seg_result
import nibabel as nib
from pathlib import Path
import shutil
from skimage.metrics import peak_signal_noise_ratio as psnr

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Brain_Env:

    def __init__(self, data_loader, budget, observation_space, device='cuda'):

        # Environment properties and initialization
        self.state = 0
        self.done = False
        self.data_loader = data_loader
        self.data_loader_iter = iter(self.data_loader)
        self.observation_space = observation_space
        self.action_space = Namespace(n=128)
        self.act_dim = self.action_space.n
        self.device = device
        self.budget = budget

    def factory_reset(self):
        self.data_loader_iter = iter(self.data_loader)

    def reset(self):
        # Reset data iterator if needed
        try:
            batch = next(self.data_loader_iter)
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            batch = next(self.data_loader_iter)

        # Move batch data to the designated device
        batch["kspace_fully"] = batch['kspace_fully'].permute(3, 0, 1, 2).to(self.device)
        batch["target"] = batch['target'].float().permute(3, 0, 1, 2).to(self.device)
        self.accumulated_mask = batch["initial_mask"].repeat(batch["target"].shape[0],1).float().unsqueeze(1).unsqueeze(-1).to(self.device)
        # batch["seg_label"] = batch["seg_label"].float().unsqueeze(1).to(self.device)

        self.state = batch
        self.num_cols = self.state['kspace_fully'].shape[-2]
        self.counter = 0
        self.done = torch.zeros(self.state['kspace_fully'].shape[0])

        # Return the masked k-space as initial observation
        s0 = self.state['kspace_fully'] * self.accumulated_mask
        return s0
    def get_state(self):
        return self.state

    def get_cur_mask_2d(self):
        cur_mask = ~self.accumulated_mask.bool()
        cur_mask = cur_mask.squeeze()
        return cur_mask

    def get_remain_epi_lines(self):
        return self.budget - self.counter

    def set_budget(self, num_lines):
        self.budget = num_lines

    def reach_budget(self):
        return self.counter >= self.budget

    def get_accumulated_mask(self):
        return self.accumulated_mask

    def step(self, action):
        info = {}
        action = torch.nn.functional.one_hot(action.long(), self.num_cols).unsqueeze(-1).unsqueeze(1)
        self.accumulated_mask = torch.max(self.accumulated_mask, action)
        self.counter += 1
        # Get observation and reward
        state = self.get_state()
        observation = state['kspace_fully'] * self.accumulated_mask
        if self.reach_budget():
            before_recons = fastmri.complex_abs(fastmri.ifft2c(torch.stack((observation.real, observation.imag), dim=-1)))
            with torch.no_grad():
                after_recons = self.recon_model(observation.squeeze(1), self.accumulated_mask.squeeze(1).unsqueeze(-1))

            ssim_before_recon = py_msssim(self.state['target'], before_recons, data_range=1.0, size_average=True).squeeze(-1)
            ssim_after_recon = py_msssim(self.state['target'], after_recons, data_range=1.0, size_average=True).squeeze(-1)
            psnr_before_recon = psnr(self.state['target'].cpu().numpy(), before_recons.cpu().numpy(), data_range=1.0)
            psnr_after_recon = psnr(self.state['target'].cpu().numpy(), after_recons.cpu().numpy(), data_range=1.0)

            img_nib = nib.load(self.state['path_img'][0])
            img_affine, img_header = img_nib.affine, img_nib.header
            save_img_path_1 = self.state['path_img'][0].replace('debug/test', 'save_RL_result/before_recons')
            save_img_path_2 = self.state['path_img'][0].replace('debug/test', 'save_RL_result/after_recons')
            save_label_path_1 = self.state['path_label'][0].replace('debug/test', 'save_RL_result/before_recons')
            save_label_path_2 = self.state['path_label'][0].replace('debug/test', 'save_RL_result/after_recons')

            save_folder_path_1 = Path(self.state['folder_path'][0].replace('debug/test', 'save_RL_result/before_recons'))
            save_folder_path_2 = Path(self.state['folder_path'][0].replace('debug/test', 'save_RL_result/after_recons'))
            if not os.path.exists(save_folder_path_1):
                os.makedirs(save_folder_path_1)
            if not os.path.exists(save_folder_path_2):
                os.makedirs(save_folder_path_2)

            nib.save(nib.Nifti1Image(before_recons.squeeze(1).permute(1, 2, 0).cpu().numpy(), img_affine, img_header), save_img_path_1)
            nib.save(nib.Nifti1Image(after_recons.squeeze(1).permute(1, 2, 0).cpu().numpy(), img_affine, img_header), save_img_path_2)
            shutil.copy(self.state['path_label'][0], save_label_path_1)
            shutil.copy(self.state['path_label'][0], save_label_path_2)

            done = torch.ones(1)  # Single done flag as budget is reached
            info['ssim_before_recons'] = ssim_before_recon
            info['ssim_after_recons'] = ssim_after_recon
            info['psnr_before_recons'] = psnr_before_recon
            info['psnr_after_recons'] = psnr_after_recon
            observation = self.reset()
        else:
            done = torch.zeros(1)  # Single done flag as budget not yet reached

        return observation, done, info


if __name__ == "__main__":
    pass
