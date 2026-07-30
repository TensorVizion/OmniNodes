# ComfyUI_WebAPI_Nodes/node_oauth_manager.py
import requests
import time
import json
from typing import Tuple

class OAuth2TokenManagerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "token_url": ("STRING", {"default": "https://oauth2.googleapis.com/token"}),
                "client_id": ("STRING", {"default": "your_client_id"}),
                "client_secret": ("STRING", {"default": "your_client_secret"}),
                "grant_type": (["client_credentials", "password"],),
            },
            "optional": {
                "scope": ("STRING", {"default": ""}),
                "username": ("STRING", {"default": ""}),
                "password": ("STRING", {"default": ""}),
                "timeout": ("FLOAT", {"default": 10.0}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("access_token", "refresh_token", "expires_at", "token_type")
    FUNCTION = "get_token"
    CATEGORY = "WebAPI Nodes"
    DESCRIPTION = "Obtain OAuth2 access token using client credentials or password flow."

    def get_token(self, token_url: str, client_id: str, client_secret: str,
                  grant_type: str, scope: str = "", username: str = "",
                  password: str = "", timeout: float = 10.0):

        try:
            data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": grant_type,
            }

            if scope:
                data["scope"] = scope
            if grant_type == "password":
                data.update({
                    "username": username,
                    "password": password,
                })

            resp = requests.post(token_url, data=data, timeout=timeout)
            resp.raise_for_status()
            tok = resp.json()

            access_token = tok.get("access_token", "")
            refresh_token = tok.get("refresh_token", "")
            expires_in = tok.get("expires_in", 3600)
            token_type = tok.get("token_type", "Bearer")
            expires_at = str(int(time.time()) + expires_in)

            return (access_token, refresh_token, expires_at, token_type)
        except Exception as e:
            return ("", "", "", f"Error: {str(e)}")

# --- Define mappings for OmniNodes ---
NODE_CLASS_MAPPINGS = {
    "OAuth2TokenManagerNode": OAuth2TokenManagerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OAuth2TokenManagerNode": "OAuth2 Token Manager (WebAPI)"
}
# ---