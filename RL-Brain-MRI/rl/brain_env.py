import fastmri
import torch
import numpy as np
from utils.cal_ssim import ssim as py_msssim
import matplotlib.pyplot as plt
# from segment.seg_test import get_seg_result

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
        batch["kspace_fully"] = batch['kspace_fully'].unsqueeze(1).to(self.device)
        batch["target"] = batch['target'].unsqueeze(1).to(self.device)
        self.accumulated_mask = batch["initial_mask"].float().unsqueeze(1).unsqueeze(-1).to(self.device)
        batch["seg_label"] = batch["seg_label"].float().unsqueeze(1).to(self.device)

        self.state = batch
        self.num_cols = self.state['kspace_fully'].shape[-2]
        self.counter = 0
        self.done = torch.zeros(self.state['kspace_fully'].shape[0])

        # Return the masked k-space as initial observation
        s0 = self.state['kspace_fully'] * self.accumulated_mask
        with torch.no_grad():
            initial_recons = self.recon_model(s0.squeeze(1), self.accumulated_mask.squeeze(1).unsqueeze(-1))
        self.previous_ssim_score = py_msssim(self.state['target'], initial_recons, data_range=1.0, size_average=False).squeeze(-1)
        return s0
    def get_state(self):
        return self.state

    def get_reward(self, observation):
        with torch.no_grad():
            recons = self.recon_model(observation.squeeze(1), self.accumulated_mask.squeeze(1).unsqueeze(-1))
        ssim_score = py_msssim(self.state['target'], recons, data_range=1.0, size_average=False).squeeze(-1)
        delta_ssim = ssim_score - self.previous_ssim_score  # Incremental improvement in SSIM
        reward = delta_ssim
        self.previous_ssim_score = ssim_score
        return torch.tensor(reward, device=self.device)

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
        reward = self.get_reward(observation)
        if self.reach_budget():
            done = torch.ones(1)  # Single done flag as budget is reached
            info['ssim_score'] = self.previous_ssim_score  # Optionally log SSIM score
            info['final_mask'] = self.accumulated_mask.clone().cpu().numpy()  # Optionally log mask
            observation = self.reset()
        else:
            done = torch.zeros(1)  # Single done flag as budget not yet reached

        return observation, reward, done, info


if __name__ == "__main__":
    pass
