# test_midas_host.py
import torch
import sys
import os
import traceback # Import traceback for detailed error messages

# --- IMPORTANT: MODIFY THESE PATHS FOR YOUR HOST MACHINE ---
# This should point to the PARENT of the 'midas' subdirectory itself
# For example, if MiDaS_v3.1_contents contains the 'midas' folder, 'run.py', etc.
midas_library_path = "/home/leonie/AI4R_RELBot/MiDaS_v3.1_contents" # Path to your MiDaS v3.1 library on host

# This should be the full path to your downloaded .pt weights file on your host
# Using the path you confirmed exists:
weights_file_on_host = "/home/leonie/AI4R_RELBot/MiDaS_v3.1_contents/weights/dpt_swin2_tiny_256.pt"
# If you prefer to use the one in your ROS workspace source (ensure it's the actual .pt file, not a symlink to a non-existent target):
# weights_file_on_host = "/home/leonie/AI4R_RELBot/ai4r_ws/src/relbot_video_interface/weights/dpt_swin2_tiny_256.pt"
# --- END IMPORTANT PATHS ---

sys.path.insert(0, midas_library_path)

print(f"Python sys.path after insert: {sys.path}")
print(f"Attempting to import DPTDepthModel from MiDaS library at: {midas_library_path}")
try:
    from midas.dpt_depth import DPTDepthModel
    print(f"Successfully imported DPTDepthModel from: {DPTDepthModel.__module__}")
except Exception as e:
    print(f"Failed to import DPTDepthModel: {e}")
    traceback.print_exc()
    exit()

print(f"Attempting to load weights from: {weights_file_on_host}")
if not os.path.exists(weights_file_on_host):
    print(f"ERROR: Weights file not found at the specified host path: {weights_file_on_host}")
    print("Please ensure the 'weights_file_on_host' variable in this script is set correctly to a valid path on your host machine.")
    exit()

print(f"Using backbone: swin2t16_256")
# Store the original torch.load
original_torch_load = torch.load

try:
    model = DPTDepthModel(
        path=None, # We are loading state_dict manually
        backbone="swin2t16_256", # This is the correct string for dpt_swin2_tiny_256.pt
        non_negative=True
    )
    print("DPTDepthModel instantiated successfully.")

    # The _load_no_safe logic
    def _load_no_safe_local(*args, **kwargs_local):
        kwargs_local.pop("weights_only", None)
        try:
            return original_torch_load(*args, weights_only=False, **kwargs_local)
        except TypeError as te:
             if "weights_only" in str(te):
                 kwargs_local.pop("weights_only", None)
                 return original_torch_load(*args, **kwargs_local)
             else:
                 raise te

    torch.load = _load_no_safe_local

    print(f"Loading state_dict from: {weights_file_on_host}")
    state_dict = torch.load(weights_file_on_host, map_location=torch.device('cpu'))
    print(f"State_dict loaded from file. Type: {type(state_dict)}")

    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            print("Found 'state_dict' key in checkpoint, using its value.")
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            print("Found 'model' key in checkpoint, using its value.")
            state_dict = state_dict["model"]
        else:
            print("Checkpoint is a dict, but no 'state_dict' or 'model' key found. Using the dict directly.")
    else:
        print(f"Checkpoint is not a dict (type: {type(state_dict)}). This is unusual for MiDaS .pt files.")

    print("Attempting to load state_dict into model...")
    model.load_state_dict(state_dict)
    print("Successfully loaded state_dict into model!")
    print("MiDaS model loaded successfully in isolated script on host.")

except Exception as e:
    print(f"ERROR during model loading in isolated script on host: {e}")
    traceback.print_exc()
finally:
    torch.load = original_torch_load
    print("Restored original torch.load.")
