import argparse
import json
import os
import copy
from functools import partial

import nibabel as nib
import numpy as np
import torch
from monai import transforms
from monai import data
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type
import matplotlib.pyplot as plt
from monai.transforms.lazy.functional import apply_pending_transforms

def dice_score(pred_mask, target_mask):
    intersection = np.logical_and(pred_mask, target_mask)
    return 2. * intersection.sum() / (pred_mask.sum() + target_mask.sum())

def load_5segmodels(args):
    pretrained_dir = args.pretrained_dir
    seg_checkpoint_name = 'model.pt'

    # Initialize the base model
    model = SwinUNETR(
    img_size=(128, 128, 128),
    in_channels=4,
    out_channels=3,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=True,
    ).to(args.device)

    # Function to load a fold model
    def load_fold_model(fold_id):
        fold_model = copy.deepcopy(model)
        fold_pretrained_pth = os.path.join(
            pretrained_dir, f'fold{fold_id}_f48_ep300_4gpu_dice0_{["8854", "9059", "8981", "8924", "9035"][fold_id]}',
            seg_checkpoint_name
        )
        fold_model_dict = torch.load(fold_pretrained_pth)["state_dict"]
        fold_model.load_state_dict(fold_model_dict)
        fold_model.eval()
        fold_model.to(args.device)
        fold_model_inferer_test = partial(
            sliding_window_inference,
            roi_size=[args.roi_x, args.roi_y, args.roi_z],
            sw_batch_size=1,
            predictor=fold_model,
            overlap=args.infer_overlap,
        )
        return fold_model_inferer_test

    # Load all fold models into a dictionary
    models_dict = {f'fold{fold_id}': load_fold_model(fold_id) for fold_id in range(5)}
    return models_dict


def get_result_prob(seg_model, input_img):
    prob = seg_model['fold0'](input_img)
    prob += seg_model['fold1'](input_img)
    prob += seg_model['fold2'](input_img)
    prob += seg_model['fold3'](input_img)
    prob += seg_model['fold4'](input_img)
    prob /= len(seg_model)
    return prob

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.6, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--pretrained_dir",default="/home/ruru/Documents/work/journal3/segmentation/pretrained_models")
parser.add_argument("--json_file", default='/home/ruru/Documents/work/journal3/segmentation/jsons/dataset_0.json')

if __name__ == "__main__":
    args = parser.parse_args()
    output_directory = "/home/ruru/Documents/work/journal3/segmentation/outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    seg_model = load_5segmodels(args)

    with open(args.json_file) as f:
        json_data = json.load(f)['testing']


    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    test_ds = data.Dataset(data=json_data, transform=test_transform)

    test_loader = data.DataLoader(
        test_ds,
        batch_size=200,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    roi = (128, 128, 128)
    model = SwinUNETR(
        img_size=roi,
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(args.device)
    root_dir = '/home/ruru/Documents/work/journal3/segmentation/pretrained_models/fold0_f48_ep300_4gpu_dice0_8854'
    model.load_state_dict(torch.load(os.path.join(root_dir, "model.pt"))["state_dict"])
    model.to(args.device)
    model.eval()

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=200,
        predictor=model,
        overlap=0.6,
    )

    with torch.no_grad():
        dices = []
        dices0 = []
        dices1 = []
        dices2 = []
        for batch_data in test_loader:
            image = batch_data["image"].cuda()
            prob = torch.sigmoid(get_result_prob(seg_model, image))
            seg = prob.detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            # seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            # seg_out[seg[1] == 1] = 2
            # seg_out[seg[0] == 1] = 1
            # seg_out[seg[2] == 1] = 4
            target_label = batch_data["label"].numpy().astype(np.int8)
            dice = dice_score(seg, target_label)
            dice0 = dice_score(seg[:, 0], target_label[:, 0])
            dice1 = dice_score(seg[:, 1], target_label[:, 1])
            dice2 = dice_score(seg[:, 2], target_label[:, 2])
            dices.append(dice)
            dices0.append(dice0)
            dices1.append(dice1)
            dices2.append(dice2)
        print(f'avg_dice:{np.array(dices).mean()}, dice0:{np.array(dices0).mean()}, dice1:{np.array(dices1).mean()}, dice2:{np.array(dices2).mean()}')

