"""
TensorVizion ComfyUI Nodes
clip_text_encode_simple_node.py — The plain positive/negative prompt-to-
conditioning node every workflow needs. CLIP Text Compare and CLIP Text
Weight already exist in this pack for analytical/weighted use cases, but
neither is a basic "just encode this text" node — this fills that gap. A
thin, TensorVizion-branded wrapper around ComfyUI's own core
CLIPTextEncode, so tokenization/embedding-resolution behaviour (including
`embedding:name` syntax from Embedding Helper) is guaranteed identical to
the standard node.
"""

from nodes import CLIPTextEncode as _CoreCLIPTextEncode


class CLIPTextEncodeSimpleNode:
    """
    Delegates straight to ComfyUI's own core `CLIPTextEncode.encode()` —
    same tokenizer, same embedding resolution, same behaviour as the
    standard node — with an added `is_positive` label purely for your own
    graph readability (it doesn't change the encoding itself; CONDITIONING
    has no inherent positive/negative flag until a sampler treats it as
    one) and a `summary` reporting a short preview of what was encoded.
    """

    CATEGORY = "TensorVizion/Prompt"

    @classmethod
    def INPUT_TYPES(cls):
        base = _CoreCLIPTextEncode.INPUT_TYPES()
        base["required"]["is_positive"] = ("BOOLEAN", {"default": True})
        return base

    RETURN_TYPES  = ("CONDITIONING", "STRING")
    RETURN_NAMES  = ("conditioning",  "summary")
    FUNCTION      = "run"

    def run(self, clip, text, is_positive):
        core = _CoreCLIPTextEncode()
        result = core.encode(clip, text)
        conditioning = result[0]

        label = "positive" if is_positive else "negative"
        preview = text.strip().replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."

        summary = f"[{label}] {preview}" if preview else f"[{label}] (empty prompt)"

        return (conditioning, summary)


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeSimpleNode": CLIPTextEncodeSimpleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeSimpleNode": "CLIP Text Encode (Simple) ✍️",
}
