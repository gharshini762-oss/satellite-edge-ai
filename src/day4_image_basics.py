import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results", exist_ok=True)
print("=== 1. NUMPY ARRAY BASICS ===")
demo_img = np.zeros((300, 400, 3), dtype=np.uint8)

demo_img[:, :, 0] = 200  # Blue background (water)
demo_img[50:250, 80:320] = [30, 180, 40]  # Green rectangle (land)
demo_img[90:170, 140:260] = [20, 40, 240]  # Bright Red square (urban/hotspot)

print("Array .shape :", demo_img.shape, "-> (Height, Width, Channels)")
print("Array .dtype :", demo_img.dtype)
print("Min value    :", demo_img.min(), "| Max value:", demo_img.max())

cv2.imwrite("results/test_photo.png", demo_img)


print("\n=== 2. OPENCV BGR TO RGB CONVERSION ===")
img_bgr = cv2.imread("results/test_photo.png")
print("Loaded with OpenCV (BGR) shape:", img_bgr.shape)
print("Center pixel in BGR:", img_bgr[130, 200])  # [Blue=20, Green=40, Red=240]


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print("Center pixel in RGB:", img_rgb[130, 200])  # [Red=240, Green=40, Blue=20]
print("\n=== 3. RESIZE TO 224x224 & SAVE ===")
img_resized_rgb = cv2.resize(img_rgb, (224, 224))
print("Resized shape:", img_resized_rgb.shape)


img_resized_bgr = cv2.resize(img_bgr, (224, 224))
cv2.imwrite("results/resized_224x224.png", img_resized_bgr)
print("Saved resized image to: results/resized_224x224.png")


print("\n=== 4. 12-BIT (0-4095) TO 8-BIT (0-255) RESCALING ===")

sat_12bit = np.linspace(0, 4095, 224 * 224, dtype=np.uint16).reshape(224, 224)
print(
    f"12-bit satellite array -> dtype: {sat_12bit.dtype}, min:"
    f" {sat_12bit.min()}, max: {sat_12bit.max()}"
)


sat_8bit = np.round((sat_12bit.astype(np.float32) / 4095.0) * 255.0).astype(
    np.uint8
)
print(
    f"Rescaled 8-bit array   -> dtype: {sat_8bit.dtype},  min:"
    f" {sat_8bit.min()}, max: {sat_8bit.max()}"
)

reconstructed_12bit = np.round(
    (sat_8bit.astype(np.float32) / 255.0) * 4095.0
).astype(np.uint16)
max_quant_error = np.abs(
    sat_12bit.astype(np.int32) - reconstructed_12bit.astype(np.int32)
).max()
print(
    "Reconstructed 12-bit   -> dtype:"
    f" {reconstructed_12bit.dtype}, min:"
    f" {reconstructed_12bit.min()}, max:"
    f" {reconstructed_12bit.max()}"
)
print(
    "Max precision loss from 12-bit -> 8-bit compression:"
    f" {max_quant_error} intensity levels"
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Plot 1: What happens if you forget to convert BGR to RGB (Red square looks Blue!)
axes[0].imshow(img_bgr)
axes[0].set_title("1. Wrong: Raw OpenCV (BGR)\nRed hotspot looks Blue!")

# Plot 2: Proper RGB conversion
axes[1].imshow(img_rgb)
axes[1].set_title("2. Correct: Converted to RGB\n(300x400 original)")

# Plot 3: Resized to 224x224
axes[2].imshow(img_resized_rgb)
axes[2].set_title("3. Resized to 224x224\n(Ready for PyTorch)")

plt.tight_layout()
plt.savefig("results/day4_output.png", dpi=150)
print("\nSaved comparison plot to: results/day4_output.png")
plt.show()