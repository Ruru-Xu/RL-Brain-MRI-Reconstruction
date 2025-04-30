import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import fastmri
import pathlib
from numpy.fft import fftshift, ifftshift, fftn, ifftn
import random

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
    kspace = fftshift(fftn(ifftshift(image, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
    return kspace / np.sqrt(np.prod(image.shape[-2:]))


def transform_kspace_to_image(kspace: np.ndarray) -> np.ndarray:
    """
    Compute the inverse Fourier transform from k-space to image space.
    """
    image = fftshift(ifftn(ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
    return np.abs(image) * np.sqrt(np.prod(kspace.shape[-2:]))

# --------- Data Loading Class ---------
class MRIDataset(Dataset):
    def __init__(self, data_path):
        self.examples = []
        self.modality = "t1ce" #["flair", "t1", "t1ce", "t2"]
        self.load_data(data_path)
    def load_data(self, base_root: pathlib.Path):
        patient_folders = sorted([f for f in base_root.iterdir() if f.is_dir()])
        for folder in patient_folders:
            patient_id = folder.name
            modality_path = folder / f"{patient_id}_{self.modality}.nii.gz"
            slices = nib.load(modality_path).shape[-1]

            numbers = [1, 3, 5, 7, 9, 11, 13, 14, 15, 16]
            for slice_idx in range(slices):
                if 'train' in str(base_root):
                    number_random = random.sample(numbers, 1)
                else:
                    number_random = random.sample(numbers, 3)
                for sampling_lines in number_random:
                    self.examples.append((modality_path, slice_idx, sampling_lines))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        modality_path, slice_idx, sampling_lines = self.examples[idx]
        modalities_normalized = nib.load(modality_path).get_fdata()[:, :, slice_idx].astype(np.float32)
        # modalities_normalized = normalize_image(center_crop_img(modalities_data))
        fully_kspace = torch.tensor(transform_image_to_kspace(modalities_normalized).astype(np.complex64))

        initial_mask = np.zeros(modalities_normalized.shape, dtype=np.float32)
        initial_mask[56:72] = 1
        sampling_mask = torch.tensor(get_sampling_mask(initial_mask, sampling_lines))

        masked_kspace = (fully_kspace * sampling_mask).unsqueeze(-1)
        masked_kspace = torch.cat((masked_kspace.real, masked_kspace.imag), dim=-1)
        undersampled_img = fastmri.ifft2c(masked_kspace)
        parameters = {
            "target": modalities_normalized,
            "input": undersampled_img,
            "sampling_mask": sampling_mask,
            'masked_kspace': masked_kspace
        }
        return parameters


def create_data_loader(args, partition, shuffle):
    dataset = MRIDataset(data_path=pathlib.Path(os.path.join(args.data_root, partition)))
    if partition == 'train':
        batch_size = args.batch_size
    else:
        batch_size = args.val_batch_size
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader

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


def get_sampling_mask(initial_mask, sampling_lines):
    # Find indices where the value is 0
    zero_indices = np.where(initial_mask[:, 0] == 0)[0]
    # Randomly select "sampling_lines" indices from the zero indices
    selected_indices = np.random.choice(zero_indices, size=sampling_lines, replace=False)
    # Create a copy of the initial mask to avoid modifying it directly
    new_mask = initial_mask.copy()
    # Set the selected indices to 1
    new_mask[selected_indices] = 1
    return new_mask