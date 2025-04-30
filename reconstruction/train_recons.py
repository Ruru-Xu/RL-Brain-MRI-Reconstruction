import argparse
import logging
import pathlib
import shutil
import time

import fastmri
import numpy as np
import torch
from torch.nn import functional as F
from models.cascade_network import build_reconstruction_model, load_recon_model
from datasets.data_loading import create_data_loader
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_epoch(args, epoch, model, data_loader, optimizer):
    model.train()
    avg_loss = 0.
    true_avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    for iter, data in enumerate(data_loader):
        batch = data
        batch['input'] = batch['input'].permute(0, 3, 1, 2).to(args.device).float()
        batch['target'] = batch['target'].to(args.device).float()
        batch['masked_kspace'] = batch['masked_kspace'].to(args.device).float()
        batch['sampling_mask'] = batch['sampling_mask'].unsqueeze(-1).to(args.device).float()

        optimizer.zero_grad()
        recon = fastmri.complex_abs(model(batch)['input'].permute(0, 2, 3, 1))
        loss = F.l1_loss(recon, batch['target'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        true_avg_loss = (true_avg_loss * iter + loss.mean()) / (iter + 1)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} AvgLoss = {avg_loss:.4g} TrueAvgLoss = {true_avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate_loss(args, model, data_loader):
    model.eval()
    losses = []
    start = time.perf_counter()
    true_avg_loss = 0.
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            batch = data
            batch['input'] = batch['input'].permute(0, 3, 1, 2).to(args.device).float()
            batch['target'] = batch['target'].to(args.device).float()
            batch['masked_kspace'] = batch['masked_kspace'].to(args.device).float()
            batch['sampling_mask'] = batch['sampling_mask'].unsqueeze(-1).to(args.device).float()

            recon = fastmri.complex_abs(model(batch)['input'].permute(0, 2, 3, 1))
            loss = F.mse_loss(recon, batch['target'], reduction='mean')
            l1_loss = (recon - batch['target']).abs()
            true_avg_loss = (true_avg_loss * iter + l1_loss.mean()) / (iter + 1)
            losses.append(loss.item())
    return np.mean(losses), true_avg_loss, time.perf_counter() - start

def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        model, start_epoch, optimizer, best_dev_loss = load_recon_model(args, optim=True)
    else:
        model = build_reconstruction_model().to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        best_dev_loss = 1e9
        start_epoch = 0


    train_loader = create_data_loader(args, 'train', shuffle=True)
    val_loader = create_data_loader(args, 'val', shuffle=False)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer)
        dev_loss, dev_l1loss, dev_time = evaluate_loss(args, model, val_loader)
        scheduler.step()

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainL1Loss = {train_loss:.4g} DevL1Loss = {dev_l1loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='********', help='Path to the dataset')
    parser.add_argument('--val_batch_size', default=480, type=int, help='Mini batch size for validation')
    parser.add_argument('--batch_size', default=64, type=int, help='Mini batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate')  # 1e-3 in Kendall&Gal, fastMRI base
    parser.add_argument('--lr_step_size', type=int, default=3, help='Period of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.96, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=1e-5,  help='Strength of weight decay regularization')
    parser.add_argument('--report_interval', type=int, default=1000, help='Period of loss reporting')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exp_dir', type=pathlib.Path, default='********/output',help='Path where model and results should be saved')
    parser.add_argument('--resume', default=True, help='If set, resume the training from a previous model checkpoint. --recon_model_checkpoint" should be set with this')
    parser.add_argument('--recon_model_checkpoint', default=pathlib.Path('*********/best_model.pt'), help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers to use for data loading')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main(args)
