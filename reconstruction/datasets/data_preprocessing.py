import os
import nibabel as nib
import numpy as np
import pathlib


def processing_save(data_root, modalities, save_path_root):
    """
    Process and save images and labels for multiple modalities.

    Args:
        data_root (Path): Path to the root directory containing patient folders.
        modalities (list): List of modalities to process (e.g., ["flair", "t1", "t1ce", "t2"]).
        save_path_root (str): Path to save the processed files.
    """
    # Iterate through each patient folder
    patient_folders = sorted([f for f in data_root.iterdir() if f.is_dir()])
    for folder in patient_folders:
        patient_id = folder.name

        # Load the segmentation label (shared across modalities)
        label_path = folder / f"{patient_id}_seg.nii.gz"
        label_nifti = nib.load(label_path)
        label_data = label_nifti.get_fdata().astype(np.float32)

        # Process the segmentation label (keep middle slices and center crop)
        slices = label_data.shape[-1]
        label_data = label_data[:, :, slices // 4:slices // 5 * 4]
        label_data = center_crop_img(label_data)
        # label_data[np.where(label_data != 0)] = 1  # Convert to binary labels

        # Save the processed label (only once per patient)
        save_patient_dir = os.path.join(save_path_root, folder.name)
        os.makedirs(save_patient_dir, exist_ok=True)
        save_label_path = os.path.join(save_patient_dir, f"{patient_id}_seg.nii.gz")
        label_nifti_out = nib.Nifti1Image(label_data, label_nifti.affine)
        nib.save(label_nifti_out, save_label_path)
        print(f"Saved processed label for {patient_id} to {save_label_path}")

        # Process and save each modality
        for modality in modalities:
            # Define the path for the current modality
            img_path = folder / f"{patient_id}_{modality}.nii.gz"
            img_nifti = nib.load(img_path)
            img_data = img_nifti.get_fdata().astype(np.float32)

            # Process the image (keep middle slices and center crop)
            img_data = img_data[:, :, slices // 4:slices // 5 * 4]
            img_data = normalize_image(center_crop_img(img_data))

            # Save the processed image
            save_img_path = os.path.join(save_patient_dir, f"{patient_id}_{modality}.nii.gz")
            img_nifti_out = nib.Nifti1Image(img_data, img_nifti.affine)
            nib.save(img_nifti_out, save_img_path)
            print(f"Saved processed {modality} for {patient_id} to {save_img_path}")


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image slice-wise to [0, 1].
    """
    slice_min = image.min(axis=(0, 1), keepdims=True)
    slice_max = image.max(axis=(0, 1), keepdims=True)
    denominator = slice_max - slice_min
    denominator[denominator == 0] = 1  # Avoid division by zero
    return (image - slice_min) / denominator


def center_crop_img(image, crop_size=128):
    """
    Directly center crop the image to the specified size.
    Assumes the input image is larger than the crop size.
    """
    H, W, _ = image.shape  # H: height, W: width

    # Calculate cropping indices
    start_h = (H - crop_size) // 2
    start_w = (W - crop_size) // 2

    # Directly crop the image
    cropped_image = image[start_h:start_h + crop_size, start_w:start_w + crop_size]

    return cropped_image


if __name__ == '__main__':
    data_root = pathlib.Path('/home/ruru/Documents/work/BraTS/BraTs2021/test')
    save_path = '/home/ruru/Documents/work/BraTS/BraTs2021/preprocessing/test'
    modalities = ["flair", "t1", "t1ce", "t2"]  # List of modalities
    processing_save(data_root, modalities, save_path)