import argparse
import pathlib
import time
import fastmri
import numpy as np
import torch
from models.cascade_network import load_recon_model
from datasets.data_loading_test import create_data_loader
from cal_ssim import ssim as py_msssim

def testing(args, model, data_loader):
    model.eval()
    ssims = []
    ssims_before = []
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            batch = data
            batch['input'] = batch['input'].permute(0, 3, 1, 2).to(args.device).float()
            batch['target'] = batch['target'].to(args.device).float()
            batch['masked_kspace'] = batch['masked_kspace'].to(args.device).float()
            batch['sampling_mask'] = batch['sampling_mask'].unsqueeze(-1).to(args.device).float()
            before_recon = fastmri.complex_abs(batch['input'].permute(0, 2, 3, 1))
            recon = fastmri.complex_abs(model(batch)['input'].permute(0, 2, 3, 1))
            ssim = py_msssim(batch['target'].unsqueeze(1), recon.unsqueeze(1), data_range=1.0)
            ssims.append(ssim.item())
            ssim_before = py_msssim(batch['target'].unsqueeze(1), before_recon.unsqueeze(1), data_range=1.0)
            ssims_before.append(ssim_before.item())
    print(f'ssim_before_recons:{np.mean(ssims_before)},  ssim:{np.mean(ssims)}')

def main(args):
    model = load_recon_model(args, optim=False)
    test_loader = create_data_loader(args, 'test', shuffle=False)
    testing(args, model, test_loader)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='*******', help='Path to the dataset')
    parser.add_argument('--val_batch_size', default=488, type=int, help='Mini batch size for validation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--recon_model_checkpoint', default=pathlib.Path('/********/best_model.pt'), help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers to use for data loading')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main(args)
