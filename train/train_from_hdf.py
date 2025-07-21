import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from models.simclr_model import SimCLR
from datasets.cryo_slice_dataset import CryoSliceDataset
from utils.losses import nt_xent_loss
from extract_slices_from_hdf import process_hdf_folder


def main():
    parser = argparse.ArgumentParser(description="Extract slices from HDF files and train SimCLR")
    parser.add_argument("--input", default="data", help="Folder containing .hdf/.h5 files")
    parser.add_argument("--slices", default="data/output_slices", help="Folder to store extracted slices")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # === Slice Extraction ===
    process_hdf_folder(args.input, args.slices)

    # === Dataset and Model ===
    dataset = CryoSliceDataset(args.slices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLR("resnet18", out_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    losses = []

    # === Training Loop ===
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            z1 = model(x1)
            z2 = model(x2)
            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        losses.append(avg)
        print(f"Epoch {epoch:03d} | Loss: {avg:.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), Path(args.checkpoint_dir) / f"simclr_epoch{epoch}.pt")

    # === Plot Loss ===
    plt.figure()
    plt.plot(range(1, args.epochs + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("NT-Xent Loss")
    plt.title("Training Loss")
    plot_path = Path(args.checkpoint_dir) / "training_loss.png"
    plt.savefig(plot_path, dpi=150)
    print(f"📈 Loss plot saved to {plot_path}")


if __name__ == "__main__":
    main()
