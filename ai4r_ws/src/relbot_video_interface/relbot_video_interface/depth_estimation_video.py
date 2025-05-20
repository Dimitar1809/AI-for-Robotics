#!/usr/bin/env python3
"""
Depth Estimation Benchmark Script
- Reads an input MP4 video
- Runs MiDaS depth estimation on each frame
- Measures and reports average inference time and FPS
"""
import os
import time
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI

# Configuration
VIDEO_PATH = "/home/mitaka1809/AI_for_Robots/AI-for-Robotics/ai4r_ws/src/relbot_video_interface/test/relbot_video.mp4"  # <-- set your video file here
MODEL_TYPE = "DPT_Hybrid"  # "DPT_Large", "DPT_Hybrid", or "MiDaS_small"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE)
midas.to(DEVICE).eval()

# Downloaded transformations
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if MODEL_TYPE in ("DPT_Large", "DPT_Hybrid"):
    midas_transform = transforms.dpt_transform
else:
    midas_transform = transforms.small_transform

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: could not open video {VIDEO_PATH}")
    exit(1)

frame_count = 0
total_inference_time = 0.0

print("Starting depth estimation benchmark...")
start_benchmark = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Convert BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare input
    input_batch = midas_transform(img).to(DEVICE)

    # Inference timing
    t0 = time.time()
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    dt = time.time() - t0
    total_inference_time += dt

    # Optionally print per-frame timing
    print(f"Frame {frame_count}: inference time = {dt:.4f} s")

# End of video
end_benchmark = time.time()

cap.release()

# Results
avg_time = total_inference_time / frame_count if frame_count > 0 else 0
fps = frame_count / total_inference_time if total_inference_time > 0 else 0
wall_clock_fps = frame_count / (end_benchmark - start_benchmark)

print("\nBenchmark Results:")
print(f"  Total frames processed: {frame_count}")
print(f"  Total inference time: {total_inference_time:.2f} s")
print(f"  Average inference time per frame: {avg_time:.4f} s")
print(f"  Inference-only FPS: {fps:.2f}")
print(f"  Wall-clock FPS (including overhead): {wall_clock_fps:.2f}")

# Optionally save the last depth map visualization
try:
    import matplotlib.pyplot as plt
    output = prediction.cpu().numpy()
    plt.imshow(output, cmap="magma")
    plt.axis("off")
    out_img = os.path.splitext(os.path.basename(VIDEO_PATH))[0] + "_last_depth.png"
    plt.savefig(out_img, bbox_inches="tight")
    print(f"Saved last depth map to {out_img}")
except Exception:
    pass
