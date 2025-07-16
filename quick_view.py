import numpy as np
import matplotlib.pyplot as plt

# Load .npy file
img = np.load("output_slices/MDF_images_0_image_xy_0064.npy")

# Display the image
plt.imshow(img, cmap='gray')
plt.title("Cryo-ET Slice")
plt.axis('off')
plt.colorbar()
plt.savefig("slice_preview1.png", dpi=150)
print("✅ Saved preview to slice_preview.png")
