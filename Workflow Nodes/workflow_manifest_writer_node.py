"""
TensorVizion ComfyUI Nodes
workflow_manifest_writer_node.py — Writes a JSON record of the
parameters that actually produced a given output: checkpoint name,
LoRA/DoRA/LyCORIS stack (name + strength each), sampler settings, seed,
and prompt text, saved alongside the image with matching numbering. The
pack has no "what exactly produced this image" record-keeping anywhere
— every generation is otherwise undocumented once the workflow graph
closes, making it hard to reproduce or compare past runs.
"""

import os
import re
import json
import datetime


class WorkflowManifestWriterNode:
    """
    Writes one JSON file per call, capturing whatever you wire in as
    plain STRING fields — this node doesn't try to introspect the graph
    automatically (ComfyUI doesn't expose a clean, stable API for a
    node to read its own upstream graph at execution time), so each
    field here is something you explicitly connect or type in.

    `extra_notes` is a free-text field for anything not covered by the
    structured fields — a quick observation about this specific run,
    a comparison note, etc.

    Filename numbering matches Custom Folder Batch Saver's convention
    (`prefix_NNNN.json`, counter read from existing files in the target
    folder) so a manifest and its corresponding image can share the
    same number if you set matching prefixes/output_dirs on both nodes.
    """

    CATEGORY = "TensorVizion/Workflow"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_dir": ("STRING", {"default": "output/tensorvizion/manifests"}),
                "prefix": ("STRING", {"default": "run"}),
                "checkpoint_name": ("STRING", {"default": ""}),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0}),
                "sampler_name": ("STRING", {"default": ""}),
                "scheduler": ("STRING", {"default": ""}),
            },
            "optional": {
                "lora_stack_summary": ("STRING", {"default": "", "multiline": True}),
                "extra_notes": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("manifest_path", "summary")
    FUNCTION = "write"

    def write(self, output_dir, prefix, checkpoint_name, positive_prompt, negative_prompt,
              seed, steps, cfg, sampler_name, scheduler, lora_stack_summary="", extra_notes=""):
        os.makedirs(output_dir, exist_ok=True)

        # Same counter-from-existing-files convention as Custom Folder
        # Batch Saver, so repeated runs never collide and a manifest can
        # share a number with its corresponding saved image.
        existing = [f for f in os.listdir(output_dir) if f.startswith(prefix + "_")]
        pattern = re.compile(re.escape(prefix) + r"_(\d+)\.json")
        existing_nums = [int(m.group(1)) for f in existing if (m := pattern.match(f))]
        next_num = (max(existing_nums) + 1) if existing_nums else 1
        num_str = f"{next_num:05d}"

        manifest = {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "checkpoint": checkpoint_name,
            "prompt": {
                "positive": positive_prompt,
                "negative": negative_prompt,
            },
            "sampling": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
            },
            "lora_stack": lora_stack_summary,
            "notes": extra_notes,
        }

        path = os.path.join(output_dir, f"{prefix}_{num_str}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        summary = f"Manifest saved: {path}"
        return (path, summary)


NODE_CLASS_MAPPINGS = {
    "WorkflowManifestWriterNode": WorkflowManifestWriterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WorkflowManifestWriterNode": "Workflow Manifest Writer 📋",
}
