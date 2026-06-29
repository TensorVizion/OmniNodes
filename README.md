OmniNodes-ComfyUI (TensorVizion)
A collection of ComfyUI custom nodes for audio processing and model utilities.

Nodes Included
Audio Nodes
Audio Beat Detect (audio_beat_detect_node.py)
Audio Normalize (audio_normalize_node.py)
Audio Spectrogram (audio_spectrogram_node.py)
Audio Waveform (audio_waveform_node.py)
Model Utilities
CLIP Text Compare (clip_text_compare_node.py)
CLIP Text Weight (clip_text_weight_node.py)
LoRA Info (lora_info_node.py)
LoRA Stack (lora_stack_node.py)
Model Block Freeze (model_block_freeze_node.py)
Weighted Model Merge (model_merge_weighted_node.py)
Installation (ComfyUI)
Locate your ComfyUI install folder.
Copy this repo into:
ComfyUI/custom_nodes/
Restart ComfyUI.
After restart, the nodes should appear in the ComfyUI UI under the categories defined by each node.

Configuration Files
This repo also includes JSON configuration files under:

tensorvizion_node_configs/tensorvizion_configs/
These may be used by the nodes (or your UI) to provide defaults / metadata.

Usage
Add nodes from the UI:

Use the Audio Nodes group for audio feature extraction/conditioning.
Use the Model Utilities group for LoRA inspection/stacking and model operations.
Requirements
Python
ComfyUI
Any dependencies are expected to be declared in pyproject.toml (or within the node modules).
Troubleshooting
Nodes do not appear: restart ComfyUI after installing.
Import errors: ensure the repository was copied into ComfyUI/custom_nodes/ exactly (folder names matter).
If you see missing mapping variables: confirm node files define NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.
License
Add your license file here (e.g., LICENSE) and link it in this README.

Acknowledgements
TensorVizion
