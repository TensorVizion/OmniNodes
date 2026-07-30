# ComfyUI_WebAPI_Nodes/node_webhook_listener.py

class WebhookListenerNode:
    _last_payload = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "port": ("INT", {"default": 8080, "min": 1024, "max": 65535}),
                "endpoint": ("STRING", {"default": "/webhook"}),
                "auth_token": ("STRING", {"default": ""}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("JSON", "STRING", "JSON", "BOOLEAN")
    RETURN_NAMES = ("payload_json", "raw_body", "headers", "new_data")
    FUNCTION = "listen"
    CATEGORY = "TensorVizion/Web API"
    DESCRIPTION = "Starts a local HTTP server to receive webhooks (use with caution)."

    def __init__(self):
        self._last_payload = None
        self._new_data = False

    def listen(self, port: int, endpoint: str, auth_token: str, unique_id: str):
        payload = getattr(self.__class__, "_last_payload", None)
        if payload is not None:
            self._new_data = True
            self.__class__._last_payload = None
            return (
                payload.get("json", {}),
                payload.get("text", ""),
                payload.get("headers", {}),
                True
            )
        else:
            return ({}, "", {}, False)

# --- Define mappings for OmniNodes ---
NODE_CLASS_MAPPINGS = {
    "WebhookListenerNode": WebhookListenerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WebhookListenerNode": "Webhook Listener (Dummy) (WebAPI)"
}
# ---