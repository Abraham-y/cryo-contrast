# Cryo-Contrast

This repository contains a minimal setup for training a SimCLR style contrastive learning model on cryo-electron tomography data. 3‑D volumes stored as HDF5 files are converted into 2‑D slices that are then used to train the model.

## Installation

```bash
pip install -r requirements.txt
```

## Workflow

1. Place your `.hdf` or `.h5` files inside the `data/` directory.
2. Run the training pipeline which will extract slices and train the model:

```bash
python train/train_from_hdf.py --input data --slices data/output_slices --epochs 100
```

The script saves model checkpoints in `checkpoints/` and a plot of the training loss in `training_loss.png`.

## Visualising Slices

You can preview any extracted slice using:

```bash
python quick_view.py --path data/output_slices/your_slice.npy --output preview.png
```

The slice will be saved to the specified PNG file.
