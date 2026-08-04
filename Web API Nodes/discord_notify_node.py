"""
TensorVizion ComfyUI Nodes
discord_notify_node.py — Posts a text message, optionally with an
attached IMAGE, to a Discord webhook URL. The "ping me when this batch
finishes" node — pairs naturally with Workflow End as the last node in a
long unattended run.
"""

import io
import numpy as np

try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
except ImportError:
    Image = None


class DiscordNotifyNode:
    """
    Posts `message` to `webhook_url` via Discord's webhook API. If an
    `image` is connected, the FIRST frame of the batch is attached as a
    PNG (Discord webhooks attach one file per request in the simple
    multipart form this node uses — for multiple images, either call
    this node once per image or use Contact Sheet Maker upstream to
    combine them into a single image first).

    Requires a Discord webhook URL, created via a Discord server's
    Server Settings -> Integrations -> Webhooks -> New Webhook. Treat
    this URL like a password — anyone with it can post to that channel.

    Requires the `requests` package (see requirements.txt) — same
    dependency as the rest of the Web API category. If it's missing,
    returns a clear error in `summary` rather than crashing the queue.
    """

    CATEGORY = "TensorVizion/Web API"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "webhook_url": ("STRING", {"default": ""}),
                "message": ("STRING", {"default": "ComfyUI run complete.", "multiline": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "username": ("STRING", {"default": "ComfyUI"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("success", "summary")
    FUNCTION = "notify"

    def notify(self, webhook_url, message, image=None, username="ComfyUI"):
        if requests is None:
            msg = (
                "[TensorVizion] 'requests' is not installed. Install it with "
                "'pip install requests' (see OmniNodes/requirements.txt) to "
                "enable Discord Notify."
            )
            return (False, msg)

        if not webhook_url.strip():
            return (False, "No webhook_url provided.")

        payload = {"content": message}
        if username.strip():
            payload["username"] = username.strip()

        files = None
        if image is not None and Image is not None:
            try:
                first_frame = (image[0].cpu().numpy() * 255.0).astype(np.uint8)
                pil_img = Image.fromarray(first_frame)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                buf.seek(0)
                files = {"file": ("image.png", buf, "image/png")}
            except Exception as e:
                # Don't fail the whole notification just because the image
                # attachment failed to encode — still send the text message.
                payload["content"] += f"\n\n(Note: image attachment failed to encode: {e})"

        try:
            if files:
                # Discord's webhook API expects the JSON payload under a
                # "payload_json" form field when a file is also attached,
                # rather than a plain JSON body.
                resp = requests.post(webhook_url, data={"payload_json": _to_json(payload)}, files=files, timeout=15)
            else:
                resp = requests.post(webhook_url, json=payload, timeout=15)

            if 200 <= resp.status_code < 300:
                return (True, f"Notification sent (HTTP {resp.status_code})")
            else:
                return (False, f"Discord returned HTTP {resp.status_code}: {resp.text[:200]}")

        except requests.RequestException as e:
            return (False, f"Request failed: {e}")


def _to_json(payload):
    import json
    return json.dumps(payload)


NODE_CLASS_MAPPINGS = {
    "DiscordNotifyNode": DiscordNotifyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiscordNotifyNode": "Discord Notify 🔔",
}
