#!/usr/bin/env python3
"""Export a single nnUNet fold checkpoint to ONNX format."""
import sys, os, json, torch
import numpy as np

WEIGHTS_DIR = "/home/haipw/.openclaw/workspace/totalseg-cpp/models/weights"
ONNX_DIR = "/home/haipw/.openclaw/workspace/totalseg-cpp/models/onnx"
os.makedirs(ONNX_DIR, exist_ok=True)

task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 291
fold = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# Find the model directory
task_dirs = [d for d in os.listdir(WEIGHTS_DIR) if d.startswith(f"Dataset{task_id}")]
if not task_dirs:
    print(f"No weights found for task {task_id}")
    sys.exit(1)

model_dir = os.path.join(WEIGHTS_DIR, task_dirs[0], "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres")
checkpoint = os.path.join(model_dir, f"fold_{fold}", "checkpoint_final.pth")
plans_file = os.path.join(model_dir, "plans.json")
dataset_file = os.path.join(model_dir, "dataset.json")

print(f"Task {task_id}, fold {fold}")
print(f"Checkpoint: {checkpoint}")

# Load plans
with open(plans_file) as f:
    plans = json.load(f)
with open(dataset_file) as f:
    dataset = json.load(f)

config = plans["configurations"]["3d_fullres"]
patch_size = config["patch_size"]
num_classes = len(dataset["labels"])
print(f"Patch size: {patch_size}, Num classes: {num_classes}")

# Use nnUNetv2 to reconstruct the network
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

plans_manager = PlansManager(plans)
configuration_manager = plans_manager.get_configuration("3d_fullres")

# Build network - newer nnUNetv2 API
arch_class_name = configuration_manager.network_arch_class_name
arch_kwargs = dict(**configuration_manager.network_arch_init_kwargs)
arch_kwargs_req_import = configuration_manager.network_arch_init_kwargs_req_import

network = get_network_from_plans(
    arch_class_name, arch_kwargs, arch_kwargs_req_import,
    input_channels=1, output_channels=num_classes,
    allow_init=True, deep_supervision=False
)

# Load checkpoint
ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
# nnUNet stores state dict under 'network_weights'
state_dict = ckpt.get("network_weights", ckpt)
network.load_state_dict(state_dict)
network.eval()

# Export to ONNX
dummy_input = torch.randn(1, 1, *patch_size)
onnx_path = os.path.join(ONNX_DIR, f"task{task_id}_fold{fold}.onnx")

print(f"Exporting to {onnx_path}...")
with torch.no_grad():
    torch.onnx.export(
        network,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "d", 3: "h", 4: "w"},
                      "output": {2: "d", 3: "h", 4: "w"}},
        opset_version=17,
        do_constant_folding=True,
    )

print(f"ONNX model saved: {onnx_path}")
print(f"File size: {os.path.getsize(onnx_path) / 1e6:.1f} MB")

# Quick validation
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path)
out = sess.run(None, {"input": dummy_input.numpy()})
print(f"Output shape: {out[0].shape} (expected [1, {num_classes}, {patch_size[0]}, {patch_size[1]}, {patch_size[2]}])")
