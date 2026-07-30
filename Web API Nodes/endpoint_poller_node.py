"""
TensorVizion ComfyUI Nodes
endpoint_poller_node.py — Repeatedly polls a URL until a JSON field
matches an expected value (or the request times out), for async job-style
APIs (submit → poll status → get result). HTTP Request only fires once;
plenty of real APIs (image gen jobs, video render queues, long-running
batch jobs) need a poll loop instead of a single call.
"""

import time
import json

try:
    import requests
except ImportError:
    requests = None


class EndpointPollerNode:
    """
    GETs `url` repeatedly every `interval_seconds` until either:
      - the JSON field at `success_field_path` equals `success_value`, or
      - `max_wait_seconds` elapses (returns `timed_out=True`).

    Built for async job APIs: submit a job elsewhere (e.g. with HTTP
    Request), then point this node at the job's status endpoint with
    something like `success_field_path="status"` and
    `success_value="completed"`.

    `success_field_path` uses the same dot-path syntax as JSON Field
    Extractor (e.g. "data.job.status"). Leave it blank to instead treat
    any 2xx response as success — useful for endpoints that return 202
    while pending and 200 once done, with no status field to check.

    Requires the `requests` package (see requirements.txt) — same
    dependency as HTTP Request / OAuth2 Token Manager / RSS Feed Parser.
    If it's missing, returns a clear error in `summary` rather than
    crashing the queue.
    """

    CATEGORY = "TensorVizion/Web API"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://httpbin.org/get"}),
                "success_field_path": ("STRING", {"default": ""}),
                "success_value": ("STRING", {"default": "completed"}),
                "interval_seconds": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 60.0, "step": 0.5}),
                "max_wait_seconds": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 600.0, "step": 1.0}),
            },
            "optional": {
                "headers": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("JSON", "BOOLEAN", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("final_response", "timed_out", "attempts", "elapsed_seconds", "summary")
    FUNCTION = "poll"

    @staticmethod
    def _resolve_path(data, path: str):
        current = data
        for segment in path.split("."):
            if isinstance(current, list):
                current = current[int(segment)]
            elif isinstance(current, dict):
                current = current[segment]
            else:
                raise TypeError(f"cannot descend into '{segment}'")
        return current

    def poll(self, url, success_field_path, success_value, interval_seconds, max_wait_seconds, headers="{}"):
        if requests is None:
            msg = (
                "[TensorVizion] 'requests' is not installed. Install it with "
                "'pip install requests' (see OmniNodes/requirements.txt) to "
                "enable Endpoint Poller."
            )
            return ({}, True, 0, 0.0, msg)

        try:
            hdrs = json.loads(headers) if headers.strip() else {}
        except json.JSONDecodeError:
            hdrs = {}

        start = time.time()
        attempts = 0
        last_response = {}
        last_error = ""

        while True:
            attempts += 1
            elapsed = time.time() - start

            try:
                resp = requests.get(url, headers=hdrs, timeout=min(10.0, max_wait_seconds))
                try:
                    body = resp.json()
                except (json.JSONDecodeError, ValueError):
                    body = {"_raw_text": resp.text}
                last_response = body

                if success_field_path.strip():
                    try:
                        actual = self._resolve_path(body, success_field_path.strip())
                        if str(actual) == str(success_value):
                            summary = (
                                f"Success — field '{success_field_path}' == "
                                f"'{success_value}' after {attempts} attempt(s), "
                                f"{elapsed:.1f}s elapsed."
                            )
                            return (body, False, attempts, elapsed, summary)
                    except (KeyError, IndexError, TypeError, ValueError):
                        pass  # field not present yet — keep polling
                else:
                    # No field path given: any 2xx response counts as success.
                    if 200 <= resp.status_code < 300:
                        summary = (
                            f"Success — HTTP {resp.status_code} after "
                            f"{attempts} attempt(s), {elapsed:.1f}s elapsed."
                        )
                        return (body, False, attempts, elapsed, summary)

            except requests.RequestException as e:
                last_error = str(e)

            if time.time() - start >= max_wait_seconds:
                summary = (
                    f"Timed out after {attempts} attempt(s), "
                    f"{time.time() - start:.1f}s elapsed."
                )
                if last_error:
                    summary += f"\nLast request error: {last_error}"
                return (last_response, True, attempts, time.time() - start, summary)

            time.sleep(min(interval_seconds, max(0.0, max_wait_seconds - (time.time() - start))))


NODE_CLASS_MAPPINGS = {
    "EndpointPollerNode": EndpointPollerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EndpointPollerNode": "Endpoint Poller ⏳",
}
