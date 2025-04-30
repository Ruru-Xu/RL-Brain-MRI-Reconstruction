from pathlib import Path
import logging
import time
import os
from hydra.core.hydra_config import HydraConfig
from collections import deque
from rl.brain_env import Brain_Env
from data_loading.mr_datamodule import MRDataModule
import hydra
import numpy as np
import torch
import random
import warnings
warnings.filterwarnings('ignore')
from recons.testing_recons import load_recon_model

def get_data(config):
    return MRDataModule(config=config)

@hydra.main(version_base=None, config_path='configs', config_name='train_brain')
def main(cfg):
    print(f"Current working directory : {os.getcwd()}")
    print(f"hydra path:{HydraConfig.get().run.dir}")
    # run_dir = Path(HydraConfig.get().run.dir)
    # cfg.snapshot_dir = (run_dir / Path("models")).resolve()
    # cfg.env.batch_size = cfg.num_envs

    data_module = get_data(cfg.env)
    test_loader = data_module.test_dataloader()
    logging.debug(f"-----length test_loader:{len(test_loader)}-----")
    eval_envs = prepare_evaluate_envs(cfg, test_loader)
    recon_model = load_recon_model(cfg.recon_model_path).to(cfg.device)
    eval_envs.recon_model = recon_model

    ac = hydra.utils.instantiate(cfg.model, action_space=eval_envs.action_space)
    ac.to(cfg.device)

    global global_step
    global_step = 0
    load_snapshot(ac, cfg.load_from_snapshot_base_dir)

    num_line = cfg.eval_num_line
    logging.info(f"=================Eval at budget of {num_line} line=================")
    eval_envs.set_budget(num_line)
    evaluate(ac, eval_envs)


def evaluate(ac, envs):
    global global_step
    ssim_before_recon = []
    psnr_before_recon = []
    ssim_after_recon = []
    psnr_after_recon = []
    envs.factory_reset()
    obs = envs.reset()
    obs_mt = torch.tensor(envs.get_remain_epi_lines()).to('cuda')
    num_done = 0
    step = 0

    while True:
        step += 1
        with torch.no_grad():
            cur_mask = envs.get_cur_mask_2d()
            input_dict = {"kspace": obs, 'mt': obs_mt}
            action, _, _, _ = ac.get_action_and_value(input_dict, cur_mask, deterministic=True)
            # action = choose_random_action(cur_mask).to('cuda')

        with torch.no_grad():
            obs, done, info = envs.step(action)
            obs_mt = torch.tensor(envs.get_remain_epi_lines()).to('cuda')

        if done.item() == 1:
            ssim_before_recon.append(info.get('ssim_before_recons', 0.0).mean().item())
            psnr_before_recon.append(info.get('psnr_before_recons', 0.0).mean().item())
            ssim_after_recon.append(info.get('ssim_after_recons', 0.0).mean().item())
            psnr_after_recon.append(info.get('psnr_after_recons', 0.0).mean().item())

            num_done += 1
            if num_done == len(envs.data_loader):
                break


    avg_ssim_before_recon = np.mean(ssim_before_recon)
    avg_psnr_before_recon = np.mean(psnr_before_recon)
    avg_ssim_after_recon = np.mean(ssim_after_recon)
    avg_psnr_after_recon = np.mean(psnr_after_recon)
    print(f'ssim_before_recon: {avg_ssim_before_recon}, psnr_before_recon: {avg_psnr_before_recon}, ssim_after_recon: {avg_ssim_after_recon}, psnr_after_recon: {avg_psnr_after_recon}')


def load_snapshot(model, load_from_snapshot_base_dir):
    snapshot_base_dir = Path(load_from_snapshot_base_dir)
    snapshot = snapshot_base_dir / f'best_model.pt'
    if not snapshot.exists():
        return None
    logging.info(f"[Train.py] load snapshot:{snapshot}")
    model.load_state_dict(torch.load(snapshot))

def prepare_evaluate_envs(cfg, val_loader):
    observation_space = cfg.env.observation_space
    envs = Brain_Env(val_loader, budget=cfg.eval_num_line, observation_space=observation_space, device=cfg.device)
    return envs

def choose_random_action(cur_mask):
    batch_size, num_cols = cur_mask.shape
    actions = []

    # Use a single generator with a fixed random seed for reproducibility or without seed for randomness
    generator = torch.Generator()
    generator.seed()

    for batch in range(batch_size):
        # Get indices of True elements
        valid_indices = torch.nonzero(cur_mask[batch], as_tuple=False).squeeze()
        if len(valid_indices) > 0:
            # Randomly select one index
            selected_action = valid_indices[torch.randint(0, len(valid_indices), (1,), generator=generator)].item()
        else:
            # Handle edge case where no True indices exist
            selected_action = -1  # Use -1 to indicate no valid action
        actions.append(selected_action)

    return torch.tensor(actions, dtype=torch.long)


if __name__ == "__main__":
    main()
