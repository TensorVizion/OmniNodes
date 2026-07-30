# ComfyUI_WebAPI_Nodes/node_http_request.py
import requests
import json
from typing import Dict, Any, Optional, Tuple

def safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return s

class HTTPRequestNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://httpbin.org/get"}),
                "method": (["GET", "POST", "PUT", "DELETE"],),
                "headers": ("STRING", {"default": '{"Content-Type": "application/json"}', "multiline": True}),
                "body": ("STRING", {"default": "{}", "multiline": True}),
            },
            "optional": {
                "timeout": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 60.0}),
            }
        }

    RETURN_TYPES = ("JSON", "STRING", "INT", "JSON")
    RETURN_NAMES = ("response_json", "response_text", "status_code", "response_headers")
    FUNCTION = "execute"
    CATEGORY = "TensorVizion/Web API"
    DESCRIPTION = "Send HTTP request and return response data."

    def execute(self, url: str, method: str, headers: str, body: str, timeout: float = 10.0):
        try:
            hdrs = safe_json_loads(headers)
            if isinstance(hdrs, str):
                hdrs = {}
            bdy = safe_json_loads(body)
            if isinstance(bdy, str):
                bdy = None

            kwargs = {"url": url, "headers": hdrs, "timeout": timeout}

            if method == "GET":
                resp = requests.get(**kwargs)
            elif method == "POST":
                resp = requests.post(**kwargs, json=bdy)
            elif method == "PUT":
                resp = requests.put(**kwargs, json=bdy)
            elif method == "DELETE":
                resp = requests.delete(**kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")

            try:
                json_out = resp.json()
            except Exception:
                json_out = {}

            return (json_out, resp.text, resp.status_code, dict(resp.headers))
        except Exception as e:
            return ({}, f"Error: {str(e)}", 0, {})

# --- Define mappings for OmniNodes ---
NODE_CLASS_MAPPINGS = {
    "HTTPRequestNode": HTTPRequestNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HTTPRequestNode": "HTTP Request (WebAPI)"
}
# ---