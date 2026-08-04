"""
TensorVizion ComfyUI Nodes
prompt_token_counter_node.py — Estimates CLIP token count for a prompt
string, reports 77-token chunking, and flags anything that will get
truncated or padded so it can be caught before a wasted generation.
"""

import re


class PromptTokenCounterNode:
    """
    Estimates how many CLIP tokens `prompt` will consume and whether it
    will be truncated at the standard 77-token (75 usable + BOS/EOS)
    chunk boundary that most SD/SDXL CLIP encoders use.

    Uses the `transformers` CLIP tokenizer if it's installed (accurate);
    otherwise falls back to a whitespace/punctuation heuristic (roughly
    correct within a token or two for typical English prompts, which is
    enough to flag truncation risk).

    `chunk_size` lets you match encoders that concatenate multiple 75-token
    chunks (e.g. ComfyUI's default CLIP Text Encode does this automatically)
    — set the number of chunks your workflow actually uses to see whether
    you're within budget across all of them combined.
    """

    CATEGORY = "TensorVizion/Prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "chunk_size": ("INT", {"default": 1, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("INT", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("token_count", "will_truncate", "truncated_preview", "summary")
    FUNCTION = "count"

    _TOKENS_PER_CHUNK = 75  # usable tokens per chunk; BOS/EOS added by the encoder

    _tokenizer = None
    _tokenizer_checked = False

    @classmethod
    def _get_tokenizer(cls):
        if cls._tokenizer_checked:
            return cls._tokenizer
        cls._tokenizer_checked = True
        try:
            from transformers import CLIPTokenizer
            cls._tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        except Exception:
            cls._tokenizer = None
        return cls._tokenizer

    def _heuristic_tokenize(self, text):
        # Rough approximation of BPE token count: split on words and
        # punctuation, treat long words as ~1 token per 4 chars (common
        # BPE subword rate for English).
        words = re.findall(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]", text)
        count = 0
        tokens_preview = []
        for w in words:
            if w.isalpha() and len(w) > 4:
                sub = max(1, len(w) // 4)
                count += sub
                tokens_preview.extend([w] * 1)  # keep word intact for preview
            else:
                count += 1
                tokens_preview.append(w)
        return count, tokens_preview

    def count(self, prompt, chunk_size):
        budget = self._TOKENS_PER_CHUNK * chunk_size
        tokenizer = self._get_tokenizer()

        if tokenizer is not None:
            ids = tokenizer(prompt, truncation=False)["input_ids"]
            # Strip BOS/EOS that the tokenizer adds for the count that matters
            usable_ids = ids[1:-1] if len(ids) >= 2 else ids
            token_count = len(usable_ids)
            will_truncate = token_count > budget

            if will_truncate:
                kept_ids = [tokenizer.bos_token_id] + usable_ids[:budget] + [tokenizer.eos_token_id]
                truncated_preview = tokenizer.decode(kept_ids, skip_special_tokens=True).strip()
            else:
                truncated_preview = prompt.strip()

            method = "transformers CLIPTokenizer (exact)"
        else:
            token_count, tokens_preview = self._heuristic_tokenize(prompt)
            will_truncate = token_count > budget

            if will_truncate:
                # Best-effort preview: keep roughly the proportional slice of words
                ratio = budget / max(1, token_count)
                cut = max(1, int(len(tokens_preview) * ratio))
                truncated_preview = " ".join(tokens_preview[:cut])
            else:
                truncated_preview = prompt.strip()

            method = "heuristic estimate (install 'transformers' for exact CLIP token counts)"

        summary = (
            f"Token count:   {token_count}\n"
            f"Budget:        {budget} ({chunk_size} x {self._TOKENS_PER_CHUNK})\n"
            f"Will truncate: {will_truncate}\n"
            f"Method:        {method}"
        )

        return (token_count, will_truncate, truncated_preview, summary)


NODE_CLASS_MAPPINGS = {
    "PromptTokenCounterNode": PromptTokenCounterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptTokenCounterNode": "Prompt Token Counter 🔢",
}
