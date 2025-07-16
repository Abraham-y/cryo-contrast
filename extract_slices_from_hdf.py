import h5py
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path

def normalize(volume):
    return (volume - np.mean(volume)) / (np.std(volume) + 1e-5)

def extract_2d_slices(volume, stride=64, crop_size=128):
    slices = []

    for z in range(0, volume.shape[0] - crop_size + 1, stride):
        slice_xy = volume[z, :, :]
        slices.append((slice_xy, f"xy_{z:04d}"))

    for y in range(0, volume.shape[1] - crop_size + 1, stride):
        slice_xz = volume[:, y, :]
        slices.append((slice_xz, f"xz_{y:04d}"))

    for x in range(0, volume.shape[2] - crop_size + 1, stride):
        slice_yz = volume[:, :, x]
        slices.append((slice_yz, f"yz_{x:04d}"))

    return slices

def visit_datasets(hdf_file):
    datasets = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            datasets.append((name, obj))
    hdf_file.visititems(visitor)
    return datasets

def process_single_file(hdf_path, output_dir, crop_size=128, stride=64):
    file_id = Path(hdf_path).stem
    try:
        with h5py.File(hdf_path, "r") as f:
            datasets = visit_datasets(f)
            if not datasets:
                print(f"⚠️  No 3D datasets found in {file_id}. Skipping.")
                return

            for full_key, dataset in datasets:
                print(f"📂 Processing {file_id} | Dataset: {full_key} | Shape: {dataset.shape}")
                volume = dataset[()]
                volume = normalize(volume)
                slices = extract_2d_slices(volume, stride, crop_size)

                for img, label in tqdm(slices, desc=f"Saving {file_id}/{full_key}", leave=False):
                    safe_key = full_key.replace('/', '_').strip('_')
                    filename = f"{file_id}_{safe_key}_{label}.npy"
                    np.save(os.path.join(output_dir, filename), img)
    except Exception as e:
        print(f"❌ Failed to process {hdf_path}: {e}")


def process_hdf_folder(folder_path, output_dir, crop_size=128, stride=64):
    os.makedirs(output_dir, exist_ok=True)
    hdf_files = list(Path(folder_path).glob("*.hdf")) + list(Path(folder_path).glob("*.h5"))

    if not hdf_files:
        print(f"❌ No .hdf or .h5 files found in {folder_path}")
        return

    print(f"🔍 Found {len(hdf_files)} HDF5 files.")
    for hdf_file in hdf_files:
        process_single_file(str(hdf_file), output_dir, crop_size, stride)

if __name__ == "__main__":
    input_folder = "data/aged"
    output_folder = "output_slices"
    crop_size = 128
    stride = 64

    process_hdf_folder(input_folder, output_folder, crop_size, stride)
