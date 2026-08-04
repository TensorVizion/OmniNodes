"""
TensorVizion ComfyUI Nodes
prompt_list_iterator_node.py — Reads prompts from a text file (one per
line) or a folder of .txt files, and returns the Nth one based on an
index input. Pairs naturally with Batch Counter (Workflow Nodes) for its
run-index output — wire Batch Counter's index into this node's `index`
input to step through the whole list one prompt per queue run, turning
"run these 50 prompts overnight" into a wireable primitive instead of
manual copy-paste-queue.
"""

import os


class PromptListIteratorNode:
    """
    source_mode:
      single_file    — `path` points at one .txt file; each non-empty
                         line (comments starting with # are skipped) is
                         one prompt.
      folder_of_files — `path` points at a directory; each .txt file's
                         full contents (whitespace-trimmed) is one prompt,
                         files sorted alphabetically for a stable order.

    `index` selects which prompt to return — wire in Batch Counter's
    index output for automatic stepping across queue runs, or set it
    manually to preview/re-run a specific prompt.

    `wrap_around` controls what happens once index reaches the end of
    the list: True cycles back to prompt 0, False clamps to the last
    prompt and reports `is_last=True` so a downstream node (e.g.
    Conditional Gate) can stop a loop cleanly instead of repeating
    forever.
    """

    CATEGORY = "TensorVizion/Workflow"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "source_mode": (["single_file", "folder_of_files"],),
                "index": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "wrap_around": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("prompt", "total_prompts", "is_last", "summary")
    FUNCTION = "run"

    def _load_prompts(self, path, source_mode):
        if source_mode == "single_file":
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Prompt file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f]
            return [line for line in lines if line and not line.startswith("#")]

        else:  # folder_of_files
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Prompt folder not found: {path}")
            txt_files = sorted(f for f in os.listdir(path) if f.lower().endswith(".txt"))
            prompts = []
            for fname in txt_files:
                with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    prompts.append(content)
            return prompts

    def run(self, path, source_mode, index, wrap_around):
        try:
            prompts = self._load_prompts(path.strip(), source_mode)
        except FileNotFoundError as e:
            return (str(e), 0, True, f"ERROR: {e}")

        if not prompts:
            msg = f"No prompts found at {path} (source_mode={source_mode})"
            return (msg, 0, True, msg)

        total = len(prompts)
        if wrap_around:
            actual_index = index % total
        else:
            actual_index = min(index, total - 1)

        is_last = (actual_index == total - 1) and not wrap_around
        prompt = prompts[actual_index]
        summary = f"Prompt {actual_index + 1}/{total}: {prompt[:60]}{'...' if len(prompt) > 60 else ''}"
        return (prompt, total, is_last, summary)


NODE_CLASS_MAPPINGS = {
    "PromptListIteratorNode": PromptListIteratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptListIteratorNode": "Prompt List Iterator 📜",
}
