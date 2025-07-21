import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Visualise an extracted slice")
    parser.add_argument("--path", required=True, help="Path to a .npy slice")
    parser.add_argument("--output", default="slice_preview.png", help="Output PNG filename")
    args = parser.parse_args()

    img = np.load(args.path)
    plt.imshow(img, cmap="gray")
    plt.title("Cryo-ET Slice")
    plt.axis("off")
    plt.colorbar()
    plt.savefig(args.output, dpi=150)
    print(f"✅ Saved preview to {args.output}")


if __name__ == "__main__":
    main()
