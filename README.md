# Optimized K-Space Under-sampling for Brain MRI Reconstruction with Reinforcement Learning

PyTorch implementation of the article
[**"Optimized K-Space Under-sampling for Brain MRI Reconstruction with Reinforcement Learning"**](https://www.sciencedirect.com/science/article/pii/S0167865526000802).

This project learns *where* to sample in k-space (the MRI frequency domain) so that accelerated, under-sampled brain MRI can be reconstructed as accurately as possible. Instead of using a fixed under-sampling pattern, a **PPO reinforcement-learning agent** selects k-space lines step by step under a fixed sampling budget, while a pre-trained **cascade reconstruction network** turns the resulting masked k-space back into an image. Reconstruction quality (SSIM) is used as the reward signal. A downstream **Swin UNETR tumor-segmentation** model is included to evaluate the effect of the learned sampling on a clinical task.

## Pipeline overview

```
                 ┌─────────────────────────┐
 Fully-sampled   │  RL agent (PPO)          │   chosen
   k-space   ──► │  picks k-space lines     ├──► sampling mask
                 │  under a fixed budget    │
                 └───────────┬─────────────┘
                             ▼
                 ┌─────────────────────────┐
                 │ Cascade reconstruction  │ ──► reconstructed image
                 │ network (pre-trained)   │
                 └───────────┬─────────────┘
                             ▼
                 reward = SSIM(recon, ground truth)
                             │
                             ▼
            (optional) Swin UNETR tumor segmentation
```

## Repository structure

| Folder | Purpose |
| --- | --- |
| `reconstruction/` | Train and test the **cascade reconstruction network** that maps masked k-space to an image. This model is trained first and used (frozen) as part of the RL reward. |
| `RL-Brain-MRI/` | **Training** the PPO agent that learns the optimal k-space sampling mask. Configured with Hydra (`configs/train_brain.yaml`). |
| `RL-Brain-testing/` | **Evaluation** of a trained RL agent, including slice- and volume-level testing environments. |
| `segmentation/` | Downstream **Swin UNETR** brain-tumor segmentation (BraTS-21) used to assess the clinical impact of the learned sampling. |

Key sub-modules:

- `*/rl/` — PPO core (`ppo_core.py`, `ppo_core_net_mt.py`), the MRI environment (`brain_env*.py`), and network utilities.
- `*/recons/` & `reconstruction/models/` — the cascade reconstruction network and loss functions.
- `*/data_loading/` & `reconstruction/datasets/` — dataset, data-module, and preprocessing code.
- `*/utils/`, `*/configs/` — SSIM metric, helpers, and Hydra configs.

## Requirements

The code is built on PyTorch. Main dependencies (Python 3.11):

```bash
pip install torch numpy fastmri pytorch-lightning hydra-core omegaconf \
            nibabel h5py scikit-image scipy matplotlib monai joblib tensorboard
```

A CUDA-capable GPU is recommended (`device: cuda` in the configs).

## Usage

### 1. Train the reconstruction network

```bash
cd reconstruction
python train_recons.py --help        # see available arguments
python train_recons.py               # train the cascade reconstruction model
python testing.py                    # evaluate (reports SSIM before/after reconstruction)
```

### 2. Train the RL sampling agent

Edit the paths in `RL-Brain-MRI/configs/train_brain.yaml`
(`train_path`, `val_path`, `recon_model_path`, `snapshot_dir`, …), then:

```bash
cd RL-Brain-MRI
python training.py                   # Hydra entry point
# override config values from the CLI, e.g.:
python training.py budget=16 num_envs=500 optim.lr=0.0004
```

Key hyper-parameters (PPO) live in `train_brain.yaml`: sampling `budget`, `num_envs`,
`num_steps`, `gamma`, `gae_lambda`, `clip_coef`, `ent_coef`, learning rate, etc.

### 3. Test the trained agent

```bash
cd RL-Brain-testing
python testing.py
```

### 4. (Optional) Tumor segmentation

See `segmentation/README.md` and `swin_unetr_brats21_segmentation_3d_1.ipynb`.
Build the dataset index with `segmentation/jsons/get_json_file.py`, then run
`segmentation/test_seg.py`.

## Configuration notes

- Training is configured with **Hydra**; outputs and model snapshots are written
  to the directories set in `train_brain.yaml`. The placeholder paths
  (`/*******/...`) **must be replaced** with your own data and output locations.
- Images and k-space are processed at `128 × 128` resolution by default.
- SSIM is the primary reconstruction metric and RL reward.

## Citation

If you use this code, please cite the paper:

> *Optimized K-Space Under-sampling for Brain MRI Reconstruction with Reinforcement Learning.*
> Pattern Recognition Letters.
> https://www.sciencedirect.com/science/article/pii/S0167865526000802

The segmentation component builds on Swin UNETR and the BraTS-21 challenge;
see `segmentation/README.md` for the full list of references.
