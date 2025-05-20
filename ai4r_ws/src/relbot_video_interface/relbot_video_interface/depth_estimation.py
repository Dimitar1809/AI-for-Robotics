#!/usr/bin/env python3
import os
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ——— Configuration ———
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid (medium accuracy, medium speed)

# Target weights directory INSIDE your package:
weights_dir = os.path.expanduser(
    "~/AI_for_Robots/AI-for-Robotics/ai4r_ws/src/relbot_video_interface/weights"
)
os.makedirs(weights_dir, exist_ok=True)
weights_file = os.path.join(weights_dir, f"{model_type}.pth")

# ——— Load & save MiDaS weights ———
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Save out the downloaded state_dict for offline use:
torch.save(midas.state_dict(), weights_file)
print(f"MiDaS weights saved to: {weights_file}")

# ——— Prepare model for inference ———
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type in ("DPT_Large", "DPT_Hybrid"):
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# ——— Load and preprocess image ———
filename = os.path.expanduser(
    "~/AI_for_Robots/AI-for-Robotics/ai4r_ws/src/relbot_video_interface/test/ut_students.png"
)
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img).to(device)

# ——— Inference & upsampling ———
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

# ——— Query depth & save visualization ———
x, y = 150, 200
print(f"Relative depth at ({x},{y}): {output[y, x]:.4f}")

plt.imshow(output, cmap="magma")
plt.axis("off")
plt.savefig("depth_map.png", bbox_inches="tight")
print("Saved depth_map.png")
