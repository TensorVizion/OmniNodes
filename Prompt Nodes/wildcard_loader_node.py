"""
TensorVizion ComfyUI Nodes
wildcard_loader_node.py — Resolves __wildcard_name__ tokens in a prompt
against .txt wildcard files, plus inline {option_a|option_b|option_c}
dynamic-prompt syntax. Pure standard library — no dynamicprompts/Impact
Pack dependency required.
"""

import os
import re
import random

import folder_paths


def _find_wildcards_dir():
    """
    Resolves the wildcards folder in priority order:
      1. <ComfyUI root>/wildcards/          (shared convention — the same
         folder Impact Pack / Dynamic Prompts / A1111-style extensions use,
         so existing wildcard collections are picked up automatically)
      2. <this node pack>/wildcards/         (bundled fallback, auto-created
         with a couple of starter files so the node works with zero setup)
    """
    base_path = getattr(folder_paths, "base_path", None)
    if base_path:
        shared_dir = os.path.join(base_path, "wildcards")
        if os.path.isdir(shared_dir):
            return shared_dir

    # No existing shared folder found — fall back to (and seed) the
    # bundled folder inside this node pack, and actually return *that*
    # path rather than the shared one, since the shared one doesn't exist.
    local_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wildcards")
    os.makedirs(local_dir, exist_ok=True)

    starter_files = {
        "color.txt": "red\nblue\ngreen\ngolden\nmuted teal\n2::crimson\n",
        "quality.txt": "masterpiece, best quality\nhighly detailed\n3::award winning photography\n",
    }
    for name, content in starter_files.items():
        path = os.path.join(local_dir, name)
        if not os.path.isfile(path):
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception:
                pass

    return local_dir


def _list_wildcard_files(wildcards_dir):
    names = []
    if os.path.isdir(wildcards_dir):
        for root, _dirs, files in os.walk(wildcards_dir):
            for f in files:
                if f.lower().endswith(".txt"):
                    rel = os.path.relpath(os.path.join(root, f), wildcards_dir)
                    names.append(rel.replace("\\", "/")[:-4])  # strip .txt
    return sorted(names)


def _load_lines(wildcards_dir, wildcard_name):
    """Loads non-empty, non-comment lines from <wildcards_dir>/<wildcard_name>.txt"""
    path = os.path.join(wildcards_dir, wildcard_name.replace("/", os.sep) + ".txt")
    if not os.path.isfile(path):
        return None
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines


def _weighted_choice(rng, lines):
    """
    Supports an optional "N::option text" weight prefix (default weight 1).
    Falls back to uniform choice if no lines carry a weight prefix.
    """
    weighted = []
    for line in lines:
        m = re.match(r"^(\d+(?:\.\d+)?)::(.*)$", line)
        if m:
            weighted.append((float(m.group(1)), m.group(2)))
        else:
            weighted.append((1.0, line))

    total = sum(w for w, _ in weighted)
    if total <= 0:
        return rng.choice(lines)

    r = rng.uniform(0, total)
    upto = 0.0
    for w, text in weighted:
        upto += w
        if r <= upto:
            return text
    return weighted[-1][1]


_WILDCARD_TOKEN = re.compile(r"__([a-zA-Z0-9_/\-]+)__")
_INLINE_CHOICE = re.compile(r"\{([^{}]+)\}")


class WildcardLoaderNode:
    """
    Resolves wildcard syntax inside a prompt string into concrete text, so a
    prompt can be written once and vary on every run/queue.

    Two syntaxes are supported, both resolvable in the same pass:

      __name__            Pulls a random (optionally weighted) line from
                           <wildcards folder>/name.txt. Nested tokens inside
                           the chosen line are resolved recursively (e.g. a
                           line in outfit.txt can itself contain __color__).

      {a|b|c}              Inline choice directly in the prompt text, no
                           separate file needed. Nests the same way.

    Weighted lines inside a wildcard file use an "N::" prefix, e.g.:
        2::crimson
        1::teal
    (crimson is twice as likely as teal). Lines without a prefix default to
    weight 1. Blank lines and lines starting with "#" are ignored.

    `seed` makes the resolution reproducible — the same seed always expands
    the same prompt to the same result. `max_recursion` guards against
    circular wildcard references (e.g. a.txt containing __b__ and b.txt
    containing __a__) by capping how many nested levels are expanded before
    giving up and leaving any remaining token as literal text.
    """

    CATEGORY = "TensorVizion/Prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "a __color__ dress, {simple|ornate|__quality__}", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_recursion": ("INT", {"default": 6, "min": 1, "max": 20}),
            },
            "optional": {
                "wildcards_subfolder": ("STRING", {"default": "", "placeholder": "subfolder under wildcards/, blank = root"}),
            }
        }

    RETURN_TYPES  = ("STRING", "STRING", "INT")
    RETURN_NAMES  = ("text", "resolution_log", "seed_used")
    FUNCTION      = "resolve"

    def resolve(self, text, seed, max_recursion, wildcards_subfolder=""):
        base_dir = _find_wildcards_dir()
        wildcards_dir = os.path.join(base_dir, wildcards_subfolder.strip().strip("/\\")) if wildcards_subfolder.strip() else base_dir

        rng = random.Random(seed)
        picks = []
        missing = []

        def expand(s, depth):
            if depth > max_recursion:
                return s

            def repl_token(m):
                name = m.group(1)
                lines = _load_lines(wildcards_dir, name)
                if lines is None:
                    missing.append(name)
                    return m.group(0)  # leave literal so it's obvious in output
                choice = _weighted_choice(rng, lines)
                picks.append(f"{name} → {choice}")
                return expand(choice, depth + 1)

            def repl_inline(m):
                options = [o for o in m.group(1).split("|")]
                choice = rng.choice(options).strip()
                picks.append(f"{{...}} → {choice}")
                return expand(choice, depth + 1)

            # Resolve inline {a|b|c} first, then __file__ tokens, one pass
            # each, then recurse if anything new appeared.
            new_s = _INLINE_CHOICE.sub(repl_inline, s)
            new_s = _WILDCARD_TOKEN.sub(repl_token, new_s)
            return new_s

        resolved = expand(text, 0)

        if picks:
            log_lines = ["Resolved:"] + [f"  {p}" for p in picks]
        else:
            log_lines = ["No wildcard tokens found — text passed through unchanged."]
        if missing:
            log_lines.append("Missing wildcard files (left as literal text):")
            log_lines.extend(f"  __{m}__ → wildcards/{m}.txt not found" for m in missing)

        return (resolved, "\n".join(log_lines), seed)


NODE_CLASS_MAPPINGS = {
    "WildcardLoaderNode": WildcardLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WildcardLoaderNode": "Wildcard Loader 🎲",
}
