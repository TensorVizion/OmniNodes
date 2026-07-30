# OmniNodes — ComfyUI Custom Node Pack

> **By TensorVizion** · 89 node files across 9 categories · Verified against
> the actual pack contents on 2026-07-30.

A production-grade ComfyUI custom node pack covering audio processing, image
post-processing, latent space manipulation, model utilities, prompt/wildcard
tooling, sampling primitives, video processing, web API integration, and
workflow control.

Most of the pack (Audio/Image/Latent/Model/Prompt/Sampling/Video/Workflow) is
built on PyTorch, NumPy, and Pillow — all bundled with any ComfyUI install, so
no extra `pip install` is needed for those eight categories. The **Web API**
category is the one exception: three of its nodes depend on the external
`requests` library (see [Requirements](#requirements)).

---

## Installation

```bash
# Option A — Git clone (recommended)
cd ComfyUI/custom_nodes/
git clone https://github.com/TensorVizion/OmniNodes

# Option B — Manual
# Download the zip, extract, and place the OmniNodes/ folder into:
# ComfyUI/custom_nodes/OmniNodes/
```

If you plan to use the **Web API** category, also install its one external
dependency:

```bash
cd ComfyUI/custom_nodes/OmniNodes/
pip install -r requirements.txt
```

Restart ComfyUI after installing. The loader (`__init__.py`) recursively scans
every `.py` file in the pack — nodes don't need to follow a specific filename
pattern to be picked up, but each file **must** define its own
`NODE_CLASS_MAPPINGS` dict or it will be silently skipped.

Nodes appear in the node search menu under nine sub-groups: `Audio`, `Image`,
`Latent`, `Model Utilities`/`Model`, `Prompt`, `Sampling`, `Video`,
`Web API`, and `Workflow` — see [Known Quirks](#known-quirks) for why Model
and Sampling nodes are split the way they are.

---

## Node Reference

### 🎵 Audio Nodes — `TensorVizion/Audio` (11 nodes)

| Node | Summary |
|------|---------|
| **Audio Beat Detect 🥁** | Energy-based onset detection; returns beat timestamps, count, and estimated BPM. |
| **Audio Mixer 🎚️** | 4-channel stereo mixer with per-track gain, pan, and mute. |
| **Audio Normalize 🔊** | Peak or RMS normalization to a target dBFS level, with DC-offset removal and soft-clip. |
| **Audio Pitch Shift 🎼** | Phase-vocoder pitch shift, ±24 semitones, pure NumPy. |
| **Audio Reverb 🏛️** | Algorithmic (Schroeder comb + allpass) or convolution (synthetic IR) reverb. |
| **Audio Sidechain Duck 🦆** | Ducks one audio signal's level based on another's envelope (classic sidechain compression). |
| **Audio Spectrogram 🎛️** | Renders an STFT spectrogram as an IMAGE, with linear/log-power scaling and 4 colormaps. |
| **Audio Stem Splitter (Freq Band) 🍰** | Frequency-band splitter into bass/mid/high stems (not source-separation ML). |
| **Audio-to-Latent Modulator 🎧** | Converts an audio envelope into a per-frame FLOAT curve and a scaled LATENT for audio-reactive generation. |
| **Audio Transient Shaper 🥊** | Boosts or reduces the attack/sustain portions of a signal. |
| **Audio Waveform 🎵** | Renders a waveform visualization as an IMAGE. |

### 🖼️ Image Nodes — `TensorVizion/Image` (10 nodes)

| Node | Summary |
|------|---------|
| **Contact Sheet Maker 🗺️** | Tiles a batch of images into a labeled contact-sheet grid. |
| **Custom Folder Batch Saver 📁** | Saves a batch to an arbitrary directory (not ComfyUI's managed output root) with persistent zero-padded numbering. |
| **Aspect Ratio Bucket 📐** | Snaps an image to the nearest standard SD/SDXL aspect-ratio training bucket. |
| **Image Blend 🖌️** | Blends two images with selectable blend modes, ratio, strength, and optional mask. |
| **Image Color Grade 🎨** | Exposure/contrast/saturation/gamma/lift/gain/temperature/tint grading. |
| **Face Detect & Crop 🙂** | Detects faces and returns cropped outputs plus a detection mask. |
| **Image Mask Composite 🖼️** | Draws shape/effect masks (darken, brighten, blur, color) directly onto an image. |
| **Image Noise Inject 🎞️** | Adds film-grain-style noise with selectable blend mode and monochrome option. |
| **Image Sharpen & Blur 🔎** | Unsharp-mask sharpening or Gaussian blur in one node. |
| **Image Vignette & Glow ✨** | Vignette darkening plus a bloom/glow effect on bright regions. |

### 🌀 Latent Nodes — `TensorVizion/Latent` (9 nodes)

| Node | Summary |
|------|---------|
| **Latent Anomaly Mask 🚩** | Flags statistically anomalous latent regions and outputs a corrected latent + mask. |
| **Latent Blend 🌀** | Blends two latents by weighted average. |
| **Latent Channel Mixer 🎚️** | Mixes/reweights latent channels, analogous to Image Channel Mixer. |
| **Latent Interpolate 🌉** | Walks between two latents (spherical/linear interpolation). |
| **Latent Mask 🎭** | Generates rectangle/ellipse/gradient masks directly in latent space. |
| **Latent Noise Inject 🌊** | Adds controlled noise directly to a latent tensor. |
| **Latent Palette Extractor 🧬** | Extracts a signature/fingerprint summary from a latent for comparison. |
| **Latent Structure Probe 📡** | Renders a heatmap of latent activation structure. |
| **Latent Visualizer 🔬** | Renders a human-viewable preview image of raw latent channels, plus stats. |

### 🧰 Model Nodes — `TensorVizion/Model Utilities` and `TensorVizion/Model` (19 nodes)

| Node | Summary |
|------|---------|
| **Simple SDXL Loader 📀** | One-node MODEL/CLIP/VAE loader for SDXL checkpoints. |
| **Batch Folder Loader 📂** | Loads a checkpoint by name from a subfolder under ComfyUI's registered checkpoint roots. |
| **CLIP Text Compare 🔍** | Encodes two prompts and reports a similarity score between their conditioning. |
| **CLIP Text Weight ⚖️** | Applies a scalar weight multiplier to CLIP conditioning. |
| **ControlNet Loader 🕹️** | Loads a ControlNet model with a summary output. |
| **Dual Model Merger 🔀** | Merges two MODELs by weighted sum. |
| **LoRA Info Inspector 🔬** | Reports rank, alpha, and target modules of a LoRA file without loading it into a pipeline. |
| **LoRA Stack 🗂️** | Chains multiple LoRAs onto a MODEL/CLIP pair in one node. |
| **Metadata Embed 🏷️** | Embeds custom metadata into a saved file. |
| **Metadata Reader 🔖** | Reads embedded metadata back out as raw text and parsed JSON. |
| **Model Block Freeze 🧊** | Freezes specific U-Net blocks (for partial fine-tuning workflows). |
| **Model Info Inspector 🔬** | Reports key count, architecture guess, and precision of a loaded MODEL. |
| **LoRA Metadata Diff 🆚** | Compares two LoRA files' metadata and reports structural compatibility. |
| **Model Merge Weighted 🔀** | Weighted merge of two models with a single output. |
| **Quick LoRA Stacker ⚡** | Lighter/faster variant of LoRA Stack for simple single-LoRA cases. |
| **Smart Unloader 🧹** | Frees VRAM by unloading models and running garbage collection; passes any type through unchanged. |
| **Trigger Word Extractor 🏹** | Pulls a LoRA's trigger word out of a prompt and returns the cleaned remainder. |
| **Upscale Model Loader 🔭** | Loads an upscale model (ESRGAN-family, etc.) with a summary output. |
| **VAE Loader 🗝️** | Loads a VAE with a summary output. |

### 🎲 Prompt Nodes — `TensorVizion/Prompt` (11 nodes)

| Node | Summary |
|------|---------|
| **CLIP Text Encode (Simple) ✍️** | Minimal CLIP text encode with a summary output, as a lighter alternative to core CLIPTextEncode. |
| **Embedding Helper 🧷** | Helps format/insert textual-inversion embedding tokens into a prompt. |
| **Negative Prompt Presets 🚫** | Dropdown-selectable common negative-prompt blocks. |
| **Prompt Cleaner 🧹** | Strips duplicate tags, extra whitespace, and malformed weighting syntax from a prompt. |
| **Prompt Combiner ➕** | Joins multiple prompt fragments into one string with configurable separators. |
| **Prompt Random Line 🎯** | Picks a random line from a multi-line text block, seeded. |
| **Prompt Token Counter 🔢** | Counts CLIP tokens in a prompt and warns if it will be truncated. |
| **Prompt Weight Scheduler ⏳** | Produces a prompt fragment with a scheduled/varying attention weight. |
| **Wildcard List Inspector 📋** | Lists available wildcard files and their line counts. |
| **Wildcard Loader 🎲** | Loads and resolves `__wildcard__` syntax from text files, seeded. |
| **Wildcard Prompt Builder 🧩** | Assembles a full prompt from multiple wildcard categories in one node. |

### 🌡️ Sampling Nodes — `TensorVizion/Model Utilities` (4 nodes)

| Node | Summary |
|------|---------|
| **Empty Latent Image ⬜** | Creates a blank latent at a given resolution/batch size, wrapping core `EmptyLatentImage` with a summary output. |
| **Simple KSampler 🌡️** | Wraps core `KSampler` with a summary output describing the sampling run. |
| **VAE Decode 🔓** | Wraps core `VAEDecode` with a summary output. |
| **VAE Encode 🔒** | Wraps core `VAEEncode` with a summary output. |

*These are thin summary-adding wrappers around ComfyUI's own core sampling
nodes (they import `EmptyLatentImage`/`KSampler`/`VAEDecode`/`VAEEncode` from
`nodes` directly), not independent reimplementations.*

### 🎬 Video Nodes — `TensorVizion/Video` (11 nodes)

| Node | Summary |
|------|---------|
| **Video Color Match 🎨** | Matches the color grade of a video batch to a reference frame/image. |
| **Video Concat / Splice 🔗** | Joins or splices IMAGE batches (ComfyUI's "video = batch of images" convention) end-to-end. |
| **Video Frame Interpolate 🎥** | Generates in-between frames to increase apparent frame rate. |
| **Video Load 📹** | Loads a video file into an IMAGE batch, reporting source FPS and frame count. |
| **Video Loop Composer 🔂** | Builds seamless ping-pong/looping sequences from a batch. |
| **Video Motion Trail 🌌** | Adds a motion-trail/ghosting effect across frames. |
| **Video Mux Audio 🔊** | Muxes an audio track onto a saved video file. |
| **Video Save 🎬** | Encodes an IMAGE batch to MP4/WEBM/GIF via `imageio` (requires the `imageio`/`imageio-ffmpeg` packages — not bundled). |
| **Video Scene Detect 🎬** | Detects hard cuts and reports scene boundaries/timestamps. |
| **Video Speed Ramp 🐢** | Applies variable-speed time-remapping to a batch. |
| **Video Trim / Extract ✂️** | Crops a batch to a start/end frame range. |

### 🌐 Web API Nodes — `TensorVizion/Web API` (8 nodes)

The category that submits requests, waits on async jobs, extracts/builds
JSON, and saves results — a full request → poll → parse → save loop.

| Node | Summary |
|------|---------|
| **HTTP Request (WebAPI)** | Sends a GET/POST/PUT/DELETE request; returns JSON, raw text, status code, and headers. |
| **OAuth2 Token Manager (WebAPI)** | Obtains an OAuth2 access token via client-credentials or password grant. |
| **RSS Feed Parser (WebAPI)** | Fetches and parses an RSS/Atom feed into a list of entries. |
| **Webhook Listener (Dummy) (WebAPI)** | ⚠️ Does not actually start an HTTP server — see [Known Quirks](#known-quirks). |
| **JSON Field Extractor 🔎** *(new)* | Pulls one value out of nested JSON via a dot-path (e.g. `data.items.0.title`), with a fallback if the path doesn't resolve. |
| **JSON Builder 🧱** *(new)* | Assembles a JSON object from up to 4 key/value pairs (with auto type coercion) plus an optional merged JSON blob — for constructing request bodies without hand-typing JSON. |
| **Endpoint Poller ⏳** *(new)* | Repeatedly GETs a URL until a JSON field matches an expected value (or any 2xx if no field given), for async job-style APIs. Times out cleanly after `max_wait_seconds`. |
| **Response Saver 💾** *(new)* | Writes a JSON or text response to disk with the pack's standard collision-avoiding numbered-filename convention. |

#### Web API node details

**HTTP Request** — Sends `GET`/`POST`/`PUT`/`DELETE` to `url` with JSON
`headers`/`body` widgets (parsed as JSON if valid, otherwise sent empty/raw).
Returns the parsed response JSON, raw response text, HTTP status code, and
response headers. Catches all exceptions internally and returns
`status_code=0` with the error message in `response_text` rather than
crashing the queue.

**OAuth2 Token Manager** — Supports `client_credentials` and `password` grant
types against any standard OAuth2 token endpoint. Returns access token,
refresh token, a computed `expires_at` unix timestamp, and token type.
Exceptions are caught and surfaced as an `"Error: ..."` string in the
`token_type` output rather than raising.

**RSS Feed Parser** — Fetches `feed_url` and extracts `<item>` blocks via
regex (not a full XML parser), pulling `title`/`link`/`description`/`pubDate`.
`filter_keyword` optionally restricts results to entries whose title or
summary contains that keyword (case-insensitive). Regex-based parsing means
malformed or heavily-namespaced feeds may parse incorrectly — for strict RSS
compliance, a real XML parser would be more robust.

**Webhook Listener (Dummy)** — See [Known Quirks](#known-quirks) below; this
node's `port`/`endpoint`/`auth_token` inputs currently have no effect.

**JSON Field Extractor** *(new)* — Accepts either a JSON dict (e.g. from HTTP
Request's `response_json`) or a raw JSON string on the same socket, auto-
detecting which it received. `field_path` uses dot notation with numeric
segments indexing into arrays (`"data.items.2.name"`). Returns `found=False`
and the `fallback` string if the path doesn't resolve, rather than raising —
a bad path won't stop the workflow.

**JSON Builder** *(new)* — 4 key/value text-widget pairs with light type
coercion: `"true"`/`"false"` → booleans, `"null"` → JSON null, plain numeric
strings → int/float, and values starting with `{`, `[`, or a quoted string
are parsed as nested JSON. `extra_json` optionally merges in a larger
hand-written object for cases needing more than 4 fields.

**Endpoint Poller** *(new)* — GETs `url` every `interval_seconds`. If
`success_field_path` is set, polls until that dot-path resolves to
`success_value` (string-compared); if left blank, any 2xx response counts as
success. Stops and reports `timed_out=True` after `max_wait_seconds`
regardless of which mode is used. Requires `requests` — see
[Requirements](#requirements).

**Response Saver** *(new)* — Accepts JSON (dict/list, pretty-printed on save)
or a plain string (saved verbatim, or re-parsed and pretty-printed if
`format="json"` and it happens to be a JSON string) on the same `content`
socket. Uses the same `name_001`, `name_002` collision-avoidance numbering as
Video Save and Custom Folder Batch Saver, so repeat runs never overwrite a
previous save.

### 🔀 Workflow Nodes — `TensorVizion/Workflow` (6 nodes)

| Node | Summary |
|------|---------|
| **Any Switch 🔀** | Boolean-gated router for any ComfyUI type — one reusable switch instead of a type-specific one per socket type. |
| **Batch Counter 🔢** | Tracks a run count across queue executions and derives a seed from it. |
| **Timer Start ⏱️▶️** | Starts a named timer, passing any type through unchanged. |
| **Timer Stop ⏱️⏹️** | Stops a named timer and reports elapsed seconds. |
| **Conditional Gate 🚦** | Routes to one of two outputs based on a boolean condition, reporting which branch fired. |
| **Workflow End 🏁** | Terminal node that accepts up to 4 inputs of any type and produces a run summary. |

---

## Known Quirks

These are real, verified characteristics of the current pack — not bugs
introduced by this doc rewrite, but worth knowing before you build around
them:

- **Web API category naming was inconsistent until this update.** The
  original 4 Web API nodes (`node_http_request.py`, `node_oauth_manager.py`,
  `node_rss_parser.py`, `node_webhook_listener.py`) used
  `CATEGORY = "WebAPI Nodes"` — the only category in the pack not prefixed
  with `TensorVizion/`. This has been corrected to `TensorVizion/Web API` to
  match every other category.
- **Webhook Listener is a dummy/stub.** Its docstring says it "starts a local
  HTTP server," but the code never binds a port or starts a listener — it
  only checks a class-level `_last_payload` attribute that nothing in the
  file ever sets. As written, this node will always return `new_data=False`.
  Treat it as a placeholder for a future real implementation, not a working
  webhook receiver.
- **Model nodes are split across two category strings.** Most Model Nodes
  use `CATEGORY = "TensorVizion/Model Utilities"`, but four files
  (`metadata_embed_node.py`, `metadata_reader_node.py`,
  `model_lora_metadata_diff_node.py`, `trigger_word_extractor_node.py`) use
  `CATEGORY = "TensorVizion/Model"` instead — so they'll show up in a
  separate submenu from the rest of the category in the node search.
- **Sampling Nodes live in `TensorVizion/Model Utilities`, not their own
  category** despite having their own `Sampling Nodes/` folder on disk. This
  is a folder-vs-category mismatch, not a bug — the loader doesn't care what
  folder a file is in, only its `CATEGORY` string.
- **`latent_mask_node.py` had a real syntax error** (four lines each
  containing two statements with no separator, e.g.
  `x0 = int(x * W)  y0 = int(y * H)`) that made the entire file fail to
  import in every prior release. This has been fixed as part of this
  update — see the [Changelog](#changelog).

---

## Folder Structure

```
OmniNodes/
├── __init__.py                 ← recursive auto-discovery loader
├── README.md
├── requirements.txt             ← NEW: declares `requests` for Web API nodes
├── pyproject.toml
├── Model Links.md               ← creator links (CivitAI/Ko-fi/Patreon), not node docs
│
├── Audio Nodes/                  (11 files, TensorVizion/Audio)
├── Image Nodes/                  (10 files, TensorVizion/Image)
├── Latent Nodes/                  (9 files, TensorVizion/Latent)
├── Model Nodes/                  (19 files, TensorVizion/Model Utilities + TensorVizion/Model)
├── Prompt Nodes/                 (11 files, TensorVizion/Prompt)
├── Sampling Nodes/                 (4 files, TensorVizion/Model Utilities)
├── Video Nodes/                  (11 files, TensorVizion/Video)
├── Web API Nodes/                  (8 files, TensorVizion/Web API — 4 original + 4 new)
│   ├── node_http_request.py
│   ├── node_oauth_manager.py
│   ├── node_rss_parser.py
│   ├── node_webhook_listener.py
│   ├── json_field_extractor_node.py   ← new
│   ├── json_builder_node.py           ← new
│   ├── endpoint_poller_node.py        ← new
│   └── response_saver_node.py         ← new
├── Workflow Nodes/                 (6 files, TensorVizion/Workflow)
│
└── Configs/                      ← JSON schema files for select nodes
```

---

## Requirements

| Dependency | Notes |
|------------|-------|
| **ComfyUI** | Any recent version |
| **Python 3.9+** | Included with ComfyUI |
| **PyTorch** | Included with ComfyUI |
| **NumPy** | Included with ComfyUI |
| **Pillow** | Included with ComfyUI |
| **requests** | **NOT bundled with ComfyUI.** Required for HTTP Request, OAuth2 Token Manager, RSS Feed Parser, and Endpoint Poller. Install with `pip install -r requirements.txt` from the pack folder, or `pip install requests` directly. Nodes that need it check for its presence and return a clear error message (rather than crashing) if it's missing. |
| **imageio** + **imageio-ffmpeg** | Optional. Only required for Video Save's mp4/webm output (GIF works with plain `imageio`). Not bundled with ComfyUI. |

All Audio/Image/Latent processing (FFT, phase vocoder, reverb, beat
detection, color grading, blending, channel mixing) is implemented in pure
NumPy/PyTorch/Pillow. The Model Utilities nodes use ComfyUI's own
`folder_paths`, `comfy.sd`, `comfy.utils`, and `comfy.model_management`
modules, already part of any ComfyUI install.

---

## Troubleshooting

**Nodes do not appear after install**
Restart ComfyUI completely. Check the terminal for `[OmniNodes]` log lines —
`✅ Loaded` means the file registered, `⚠️ No NODE_CLASS_MAPPINGS` means the
file was found but skipped, and `❌ Error importing` means a real failure
with a traceback printed below it.

**Web API nodes fail to import with `ModuleNotFoundError: No module named 'requests'`**
Run `pip install -r requirements.txt` from inside the `OmniNodes/` folder (or
`pip install requests` directly) into the same Python environment ComfyUI
runs in, then restart ComfyUI.

**Endpoint Poller always times out**
Check `success_field_path` matches the actual shape of the JSON your endpoint
returns — a typo'd path never resolves, so the node polls until
`max_wait_seconds` and reports `timed_out=True`. Leave `success_field_path`
blank to instead succeed on any 2xx response if your endpoint has no status
field to check.

**Webhook Listener never returns new data**
This is expected — see [Known Quirks](#known-quirks). It's a dummy node that
was never wired to a real HTTP server.

**Import error on a specific node**
Read the traceback in the ComfyUI terminal. Failures here are almost always
missing ComfyUI internals (e.g. `folder_paths`, `comfy.sd`, or core nodes
like `KSampler`/`VAEDecode`) not being on the Python path, which usually
means the pack isn't actually inside `ComfyUI/custom_nodes/` — check the
install location first.

**LoRA Stack / Quick LoRA Stacker / LoRA Info Inspector say `LoRA not found`**
LoRA filenames come from ComfyUI's `folder_paths` registry. Make sure your
LoRAs are in the folder ComfyUI expects (usually `ComfyUI/models/loras/`).

**Simple SDXL Loader / Batch Folder Loader can't find a checkpoint**
Same as above — both nodes only see files under ComfyUI's registered
`checkpoints/` search paths, not arbitrary filesystem locations.

**Smart Unloader doesn't seem to free any VRAM**
Check the `summary` output string — if CUDA isn't available, it reports
"CUDA not available" and skips VRAM accounting, since there's nothing to
measure. The unload/GC calls still run either way.

---

## Changelog

**2026-07-30**
- Added 4 new Web API nodes: JSON Field Extractor 🔎, JSON Builder 🧱,
  Endpoint Poller ⏳, Response Saver 💾 — rounding out the category into a
  full request → poll → parse → save loop.
- Recategorized the original 4 Web API nodes from `CATEGORY = "WebAPI Nodes"`
  to `TensorVizion/Web API`, matching every other category's naming.
- Added `requirements.txt` declaring the `requests` dependency, previously
  undeclared anywhere in the pack.
- Fixed a real syntax error in `latent_mask_node.py` (four lines each
  containing two unseparated statements) that made the file fail to import
  in every prior release.
- Rewrote this README from scratch to reflect the pack's actual current
  contents — the previous version predated the Sampling, Video, and Web API
  categories entirely and had drifted on per-category counts elsewhere.

---

## License

MIT — see `LICENSE` file.

---

*OmniNodes by TensorVizion · github.com/TensorVizion/OmniNodes*
