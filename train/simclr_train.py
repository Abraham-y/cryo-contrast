import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models.simclr_model import SimCLR
from datasets.cryo_slice_dataset import CryoSliceDataset
from utils.losses import nt_xent_loss
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Config ===
data_path = "data/output_slices"
batch_size = 128
epochs = 100
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# === Setup ===
dataset = CryoSliceDataset(data_path)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = SimCLR("resnet18", out_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# === Training Loop ===
for epoch in range(1, epochs + 1):
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

    print(f"Epoch {epoch:03d} | Loss: {total_loss / len(loader):.4f}")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{checkpoint_dir}/simclr_epoch{epoch}.pt")
