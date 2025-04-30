import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import nibabel as nib
import fastmri
import pathlib
from numpy.fft import fftshift, ifftshift, fftn, ifftn
from typing import List, Tuple


# --------- Utility Functions ---------
def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image slice-wise to [0, 1].
    """
    slice_min = image.min(axis=(0, 1), keepdims=True)
    slice_max = image.max(axis=(0, 1), keepdims=True)
    denominator = slice_max - slice_min
    denominator[denominator == 0] = 1  # Avoid division by zero
    return (image - slice_min) / denominator


def transform_image_to_kspace(image: np.ndarray) -> np.ndarray:
    """
    Compute the Fourier transform from image space to k-space.
    """
    kspace = fftshift(fftn(ifftshift(image, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    return kspace / np.sqrt(np.prod(image.shape[:2]))


def transform_kspace_to_image(kspace: np.ndarray) -> np.ndarray:
    """
    Compute the inverse Fourier transform from k-space to image space.
    """
    image = fftshift(ifftn(ifftshift(kspace, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    return np.abs(image) * np.sqrt(np.prod(kspace.shape[:2]))

# --------- Data Loading Class ---------
class MRIDataset(Dataset):
    def __init__(self, data_path):
        self.modality = "t2" #["flair", "t1", "t1ce", "t2"]
        self.examples = []
        self.load_data(data_path)

    def load_data(self, base_root: pathlib.Path):
        patient_folders = sorted([f for f in base_root.iterdir() if f.is_dir()])
        for folder in patient_folders:
            patient_id = folder.name
            modality_path = folder / f"{patient_id}_{self.modality}.nii.gz"
            label_path = folder / f"{patient_id}_seg.nii.gz"
            self.examples.append((modality_path, label_path, folder))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset, including k-space data.
        """
        modality_path, path_label, folder_path = self.examples[idx]
        data_img = nib.load(modality_path)
        modalities_normalized = data_img.get_fdata()
        # modalities_normalized = normalize_image(center_crop_img(modalities_data))
        fully_kspace = transform_image_to_kspace(modalities_normalized).astype(np.complex64)

        # label_data = nib.load(path_label).get_fdata()

        sampling_mask = np.zeros(modalities_normalized.shape[0], dtype=np.float32)
        sampling_mask[56:72] = 1

        # Return parameters
        parameters = {
            "target": modalities_normalized,  # Normalized image-space data
            "kspace_fully": fully_kspace,  # Fully sampled k-space data
            # "seg_label": label_data,  # Multi-class segmentation mask
            "initial_mask": sampling_mask,  # Cartesian sampling mask
            "folder_path": str(folder_path),
            "path_img": str(modality_path),
            "path_label": str(path_label),
        }
        return parameters


# --------- Data Loader Creation ---------
def create_dataloader(data_path, dataset_type, batch_size=8, num_workers=4, shuffle=True):
    """Create dataloader for MRI datasets"""
    dataset = MRIDataset(data_path, dataset_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def center_crop_img(image, crop_size=128):
    """
    Directly center crop the image to the specified size.
    Assumes the input image is larger than the crop size.
    """
    H, W = image.shape  # H: height, W: width

    # Calculate cropping indices
    start_h = (H - crop_size) // 2
    start_w = (W - crop_size) // 2

    # Directly crop the image
    cropped_image = image[start_h:start_h + crop_size, start_w:start_w + crop_size]

    return cropped_image