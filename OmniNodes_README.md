# OmniNodes — ComfyUI Custom Node Pack

> **By TensorVizion** · 30 working nodes · 4 categories · Zero heavy dependencies
> (+ 6 draft nodes not yet wired up — see [Known Issues](#known-issues--draft-nodes))

A production-grade ComfyUI custom node pack covering audio processing, image
post-processing, latent space manipulation, and model utilities — built almost
entirely on PyTorch and NumPy with no external audio/image libraries required.

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

Restart ComfyUI after installing. The loader (`__init__.py`) recursively scans
every `.py` file in the pack — nodes don't need to follow a specific filename
pattern to be picked up, but each file **must** define its own
`NODE_CLASS_MAPPINGS` dict or it will be silently skipped (see
[Known Issues](#known-issues--draft-nodes)).

Working nodes appear under the **TensorVizion/** category group in the node
search menu, split into four sub-groups: `Audio`, `Image`, `Latent`, and
`Model Utilities`.

---

## Node Reference

### 🎵 Audio Nodes — `TensorVizion/Audio` (13 nodes)

| Node | Inputs | Outputs |
|------|--------|---------|
| **Audio Load 📂** | `audio_file` (dropdown from `input/`) · optional `file_path_override` | `audio` AUDIO |
| **Audio Save 💾** | `audio_samples`, `filename_prefix`, `bit_depth` (16/32) | `saved_path` STRING |
| **Audio Trim ✂️** | `audio_samples`, `start_seconds`, `end_seconds`, `duration_seconds`, `pad_if_short` | `audio` AUDIO · `duration_out` FLOAT · `summary` STRING |
| **Audio Concat ➕** | `audio_1`, `crossfade_ms`, `gap_ms`, `auto_resample` · optional `audio_2/3/4` | `audio` AUDIO · `total_duration` FLOAT · `summary` STRING |
| **Audio Fade 🌅** | `audio_samples`, `fade_in_ms`, `fade_out_ms`, `curve` | `audio` AUDIO · `summary` STRING |
| **Audio Resample 🔁** | `audio_samples`, `target_sample_rate`, `channel_mode`, `anti_alias_filter` | `audio` AUDIO · `output_sample_rate` INT · `summary` STRING |
| **Audio Normalize 🔊** | `audio_samples`, `target_db`, `normalize_mode`, `remove_dc_offset`, `soft_clip` | `audio` AUDIO · `peak_db` FLOAT · `rms_db` FLOAT · `summary` STRING |
| **Audio Mixer 🎚️** | `master_gain_db` · 4× tracks each with `audio` (optional), `gain_db`, `pan`, `mute` | `audio` AUDIO · `summary` STRING |
| **Audio Pitch Shift 🎼** | `audio_samples`, `semitones`, `fft_size`, `hop_length`, `preserve_formants` | `audio` AUDIO · `summary` STRING |
| **Audio Reverb 🏛️** | `audio_samples`, `sample_rate`, `reverb_type`, `room_size`, `damping`, `wet_level`, `dry_level`, `pre_delay_ms`, `stereo_width` | `audio` AUDIO · `summary` STRING |
| **Audio Beat Detect 🥁** | `audio_samples`, `sample_rate`, `sensitivity`, `min_bpm`, `max_bpm` | `beat_times` STRING · `beat_count` INT · `bpm` FLOAT · `summary` STRING |
| **Audio Spectrogram 🎛️** | `audio_samples`, `sample_rate`, `width`, `height`, `fft_size`, `hop_length`, `spectrogram_type`, `colormap` | `spectrogram_image` IMAGE |
| **Audio Waveform 🎵** | `audio_samples`, `width`, `height`, `line_color_r/g/b`, `bg_color_r/g/b`, `line_thickness` | `waveform_image` IMAGE |

#### Audio node details

**Audio Load** — Entry point for audio workflows. Reads uncompressed PCM WAV
(8/16/32-bit) from ComfyUI's `input/` folder using only the standard-library
`wave` module — no soundfile/librosa dependency. Pick a file from the dropdown
or override with an absolute path.

**Audio Save** — Exit point for audio workflows. Writes AUDIO to a WAV file in
`output/` using ComfyUI's standard `prefix_00001_.wav` numbering so repeat runs
never overwrite previous saves. `bit_depth` selects 16 or 32-bit PCM.

**Audio Trim** — Crops a clip to a start/end range. Leave `end_seconds` and
`duration_seconds` both at 0 to keep everything from `start_seconds` onward.
`pad_if_short` zero-pads if the requested range extends past the clip.

**Audio Concat** — Joins up to 4 clips end-to-end in order, with optional
crossfade (equal-power linear ramp) or inserted silence (`gap_ms`) between
each pair. Auto-resamples and channel-matches clips that don't match `audio_1`.
Complements Audio Mixer, which blends clips simultaneously instead of
sequentially.

**Audio Fade** — Independent fade-in/out envelopes with four curve shapes:
`linear`, `exponential` (slow start), `logarithmic` (fast start), `s_curve`
(smoothstep, eases both ends).

**Audio Resample** — Changes sample rate (e.g. 44100 → 22050) and/or channel
layout (`keep` / `to_mono` / `to_stereo`) via pure NumPy linear interpolation.
Downsampling applies a light moving-average low-pass filter first to reduce
aliasing.

**Audio Normalize** — Peak or RMS normalisation to a target dBFS level.
Optional DC offset removal runs before level detection; `soft_clip` applies a
tanh limiter at the output stage.

**Audio Mixer** — 4-channel stereo mixer. Each track has independent
`gain_db` (−60 to +12 dB), constant-power stereo `pan`, and a `mute` toggle.
Tracks are optional — leave sockets disconnected to leave slots empty. Output
is protected by a tanh soft-clipper.

**Audio Pitch Shift** — Phase-vocoder pitch shift, ±24 semitones (2 octaves),
pure NumPy. Processes each channel independently; mono inputs are handled
transparently.

**Audio Reverb** — Two modes: `algorithmic` (Schroeder comb + allpass filter
network) and `convolution` (synthetic decaying impulse response via FFT
convolution). `room_size` scales delay times and IR length; `stereo_width`
widens the mid/side image of the wet signal.

**Audio Beat Detect** — Energy-based onset detection. Returns beat timestamps
as a comma-separated string, total count, and estimated BPM clamped to
`[min_bpm, max_bpm]`. `sensitivity` sets how much louder a frame must be
relative to its local window to count as a beat (default 1.5×).

**Audio Spectrogram** — STFT-based spectrogram rendered as an IMAGE.
`spectrogram_type`: `linear` (raw magnitude) or `log_power` (log1p, perceptual
scaling). Four colormaps: `cyan_dark`, `viridis`, `hot`, `grayscale`. Frequency
axis is flipped so low frequencies sit at the bottom.

**Audio Waveform** — Renders a waveform visualisation as an IMAGE. Stereo
inputs collapse to mono by averaging. RGB is set per-channel for both the
waveform line and background.

---

### 🖼️ Image Nodes — `TensorVizion/Image` (6 nodes)

| Node | Inputs | Outputs |
|------|--------|---------|
| **Image Blend 🖌️** | `image_a`, `image_b`, `blend_mode`, `ratio`, `strength`, `clamp_output` · optional `mask` | `image` IMAGE · `blend_info` STRING |
| **Image Color Grade 🎨** | `image`, `exposure`, `contrast`, `saturation`, `gamma`, `lift`, `gain`, `temperature`, `tint` | `image` IMAGE · `grade_info` STRING |
| **Image Mask Composite 🖼️** | `image`, `mask_shape`, `effect`, `x`, `y`, `width`, `height`, `feather`, `invert_mask`, `darken_amount`, `brighten_amount`, `blur_radius`, `color_r/g/b` · optional `mask` | `image` IMAGE · `mask_out` MASK |
| **Image Noise Inject 🎞️** | `image`, `noise_type`, `blend_mode`, `strength`, `seed`, `monochrome`, `clamp_output` | `image` IMAGE |
| **Image Sharpen & Blur 🔎** | `image`, `mode`, `radius`, `strength` | `image` IMAGE |
| **Image Vignette & Glow ✨** | `image`, `vignette_amount`, `vignette_radius`, `glow_threshold`, `glow_strength`, `glow_radius` | `image` IMAGE |

#### Image node details

This category is the pixel-space counterpart to the Latent category below —
several nodes are direct analogues (Image Blend ↔ Latent Blend, Image Noise
Inject ↔ Latent Noise Inject, Image Mask Composite ↔ Latent Mask).

**Image Blend** — Blends two images with the same nine modes as Latent Blend:
`lerp`, `add`, `subtract`, `multiply`, `screen`, `overlay`, `difference`,
`hardlight`, `spatial_mask` (uses a connected MASK — A outside, B inside).
`image_b` is auto-resized to match `image_a` if resolutions differ.

**Image Color Grade** — A full lift/gamma/gain colour pipeline applied in
order: exposure (2^stops) → lift/gain → gamma → contrast (pivoted at 0.5) →
saturation → temperature (warm/cool) → tint (magenta/green). All parameters
default to a neutral no-op so you can dial in from a clean starting point.

**Image Mask Composite** — Applies one effect to a masked region of a single
image (for blending *two* images together, use Image Blend's `spatial_mask`
mode instead). Shapes: `rectangle`, `ellipse`, `gradient_h`, `gradient_v`, or
`external` (connect your own MASK). Effects: `darken`, `brighten`,
`desaturate`, `solid_color`, `invert`, `blur`.

**Image Noise Inject** — Same noise engine as Latent Noise Inject, applied to
pixels instead: `gaussian`, `uniform`, `perlin` (approximate), `salt_pepper`.
`monochrome` applies identical noise to R/G/B for a classic film-grain look;
off gives independent per-channel colour noise. `seed` makes it reproducible.

**Image Sharpen & Blur** — Three modes in one node: `sharpen` (unsharp mask),
`gaussian_blur`, `box_blur`. `radius` sets kernel size (auto-rounded to odd);
`strength` blends the effect back toward the original (1.0 = full effect).
Pure PyTorch — no PIL filters or OpenCV.

**Image Vignette & Glow** — Two effects in one node, applied in order: (1)
glow/bloom — pixels above `glow_threshold` are extracted, blurred, and added
back at `glow_strength`; (2) vignette — radial darkening from the centre,
`vignette_radius` sets how far out the falloff starts, `vignette_amount` sets
corner darkness. Set either amount to 0 to disable that half.

---

### 🌀 Latent — `TensorVizion/Latent` (5 nodes)

| Node | Inputs | Outputs |
|------|--------|---------|
| **Latent Blend 🌀** | `latent_a`, `latent_b`, `blend_mode`, `ratio`, `strength`, `normalize_output` · optional `mask` | `latent` LATENT · `blend_info` STRING |
| **Latent Channel Mixer 🎚️** | `latent`, `gains`, `offsets`, `channel_order`, `normalize_output` | `latent` LATENT |
| **Latent Interpolate 🌉** | `latent_a`, `latent_b`, `steps`, `method`, `easing`, `include_endpoints` | `latent` LATENT (batched) |
| **Latent Noise Inject 🌊** | `latent`, `noise_type`, `blend_mode`, `strength`, `seed`, `channel_mask` | `latent` LATENT |
| **Latent Visualizer 🔬** | `latent`, `layout`, `colormap`, `channel_index`, `normalize_per_channel`, `tile_size` | `image` IMAGE · stats STRING |

#### Latent node details

**Latent Blend** — Blends two latent tensors with nine modes (see Image Blend
above for the shared list). `spatial_mask` mode blends spatially using a
connected MASK. B is auto-resized to match A if spatial sizes differ.

**Latent Channel Mixer** — Per-channel gain/offset remixing plus optional
channel reordering — a latent-space analogue of an RGB channel mixer. Works
with any channel count (4 for SD1.5/SDXL, 16 for SD3/Flux) since gains/offsets
are comma-separated strings that cycle to fit however many channels the
latent actually has. `channel_order`, e.g. `"1,0,2,3"`, swaps channels 0 and 1;
leave blank to skip reordering.

**Latent Interpolate** — Generates `steps` latents walking from `latent_a` to
`latent_b`, stacked as a single batch — feed straight into a batched
KSampler/VAE Decode/Save Image chain for a morph animation. `method`: `lerp`
(fast, can dip in "energy" between very different latents) or `slerp`
(spherical, keeps roughly constant magnitude — generally the better choice for
diffusion latents). `easing` reshapes step spacing (`linear`, `ease_in`,
`ease_out`, `ease_in_out`). Note: only the first batch item of each input
latent is used.

**Latent Noise Inject** — Adds structured noise directly to a latent tensor.
Types: `gaussian`, `uniform`, `perlin` (4-octave approximate via bilinear
upsampling), `salt_pepper`. Blend modes: `add`, `multiply`, `lerp`.
`channel_mask` accepts `all` or a comma-separated channel index list (e.g.
`0,1`) to restrict injection to specific channels. `seed` makes it
reproducible.

**Latent Visualizer** — Renders a fast false-colour preview of a latent's raw
channels *without* a VAE decode — a structural/diagnostic view (composition,
contrast, whether a channel is dead or saturated), not what the final decoded
image will look like. `layout`: `grid` (tiles every channel) or
`single_channel` (shows just `channel_index`, larger). Only the first batch
item is visualised; stats are computed over the full batch.

---

### 🔧 Model Utilities — `TensorVizion/Model Utilities` (6 nodes)

| Node | Inputs | Outputs |
|------|--------|---------|
| **CLIP Text Compare 🔍** | `clip`, `prompt_a`, `prompt_b` | `conditioning_a` · `conditioning_b` · `similarity` FLOAT · `similarity_label` STRING |
| **CLIP Text Weight ⚖️** | `clip`, `prompt`, `weight` · optional `conditioning_in`, `blend_ratio` | `conditioning` CONDITIONING |
| **LoRA Info Inspector 🔬** | `lora_name` | `summary` STRING · `key_count` INT · `rank` INT · `alpha` FLOAT · `target_modules` STRING |
| **LoRA Stack 🗂️** | `model`, `clip` · 5× `lora_N`, `model_weight_N`, `clip_weight_N` | `model` MODEL · `clip` CLIP · `stack_summary` STRING |
| **Model Block Freeze 🧊** | `model`, `freeze_input/middle/output`, block range ints, `freeze_strength` | `model` MODEL · `freeze_summary` STRING |
| **Model Merge Weighted 🔀** | `model_a`, `model_b`, `ratio`, `strategy` | `merged_model` MODEL |

#### Model utility details

**CLIP Text Compare** — Encodes two prompts and computes cosine similarity on
the pooled CLIP embeddings. Returns both conditionings (ready to use
downstream) plus a `similarity` float in [0, 1] and a human-readable label.
Useful for prompt A/B testing and measuring semantic distance between
concepts.

**CLIP Text Weight** — Encodes a prompt and multiplies the conditioning
tensor by `weight`, giving a numeric slider alternative to `(prompt:1.3)`
bracket syntax. When `conditioning_in` is connected, blends the new
conditioning with the incoming one at `blend_ratio` (0 = all incoming, 1 = all
new). Supports negative weights for concept negation.

**LoRA Info Inspector** — Reads a LoRA file and returns structural metadata
without applying it to any model: key count, detected rank (from
`lora_down` tensor shape), alpha, estimated parameter count, and a list of
unique target-module prefixes. Use this to audit an unfamiliar LoRA before
loading it.

**LoRA Stack** — Applies up to 5 LoRAs in sequence with individual
`model_weight`/`clip_weight` per slot. Slots set to `None` are skipped.
`stack_summary` lists every applied LoRA with its weights. Replaces chaining
multiple Load LoRA nodes.

**Model Block Freeze** — Applies zero-weight patches to selected input,
middle, or output block ranges of a diffusion UNet. `freeze_strength` at 0.0
fully neutralises the selected blocks; at 1.0 leaves them unchanged
(effectively a no-op). Useful for style ablation and partial-inference
experiments.

**Model Merge Weighted** — Three merge strategies:
- `weighted_sum` — classic linear blend: `out = A·(1−r) + B·r`
- `sigmoid_blend` — smooth S-curve applied across layer depth; layers near
  the `ratio` boundary blend softly while distant layers snap to A or B
- `layer_select` — encoder blocks from A, decoder blocks from B, split at
  `ratio`

---

## Known Issues / Draft Nodes

`Model Nodes/` contains **6 additional files that are not yet functional** and
will not appear in ComfyUI at all in their current state:

| File | Intended purpose | Why it doesn't load |
|------|------------------|----------------------|
| `SDXL_Loader.py` (`SimpleSDXLLoader`) | Checkpoint + auto-matching VAE loader | No `NODE_CLASS_MAPPINGS`; missing `import folder_paths` and `import comfy.sd` |
| `batch_folder_loader.py` (`BatchFolderLoader`) | Load/list all checkpoints in a folder | No `NODE_CLASS_MAPPINGS`; missing `import comfy.sd`; uses a raw path string instead of ComfyUI's `folder_paths` registry |
| `dual_model_merger.py` (`DualModelMerger`) | Model merge with Weighted Sum / Add Difference / Slerp | No `NODE_CLASS_MAPPINGS`; calls `comfy.model_merging.merge_models(...)` with an `interpolation_method` kwarg that isn't part of ComfyUI's real merge API |
| `model_info_inspector.py` (`ModelInfoInspector`) | Read checkpoint metadata without loading it | No `NODE_CLASS_MAPPINGS`; missing `import folder_paths` |
| `quick_lora_stacker.py` (`QuickLoRAStacker`) | Lightweight 3-slot LoRA stacker | No `NODE_CLASS_MAPPINGS`; missing `import folder_paths` and `import comfy.sd` |
| `smart_unloader.py` (`SmartUnloader`) | Free VRAM by unloading model/VAE/CLIP | No `NODE_CLASS_MAPPINGS`; missing `import torch`; returns `(None, None, None)` for MODEL/VAE/CLIP, which will break any downstream node still expecting those types |

Because the auto-discovery loader in `__init__.py` requires each file to
define its own `NODE_CLASS_MAPPINGS` dict, these six are currently skipped
silently at startup (you'll see a `⚠️ No NODE_CLASS_MAPPINGS` line in the
console for each, not an error) — they take up space in the pack but do
nothing yet. They're kept in this build as drafts/in-progress work rather than
deleted, since three of them (`SDXL_Loader`, `QuickLoRAStacker`,
`SmartUnloader`) cover genuinely useful gaps in the current node set — see
[Where to Go Next](#where-to-go-next) below.

If you want any of these working immediately: add the missing imports,
append a `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` block matching
the pattern every other file in the pack uses, and (for `dual_model_merger.py`
specifically) replace the `comfy.model_merging.merge_models` call with logic
modeled on the working `Model Merge Weighted` node in this same pack, which
already implements a correct weighted-sum/sigmoid/layer-select merge.

---

## Folder Structure

```
OmniNodes/
├── __init__.py               ← recursive auto-discovery loader
├── README.md
├── pyproject.toml
├── Model Links.md            ← creator links (CivitAI/Ko-fi/Patreon), not node docs
│
├── Audio Nodes/               ← TensorVizion/Audio category (13 nodes)
│   ├── audio_load_node.py
│   ├── audio_save_node.py
│   ├── audio_trim_node.py
│   ├── audio_concat_node.py
│   ├── audio_fade_node.py
│   ├── audio_resample_node.py
│   ├── audio_normalize_node.py
│   ├── audio_mixer_node.py
│   ├── audio_pitch_shift_node.py
│   ├── audio_reverb_node.py
│   ├── audio_beat_detect_node.py
│   ├── audio_spectrogram_node.py
│   └── audio_waveform_node.py
│
├── Image Nodes/               ← TensorVizion/Image category (6 nodes)
│   ├── image_blend_node.py
│   ├── image_color_grade_node.py
│   ├── image_mask_composite_node.py
│   ├── image_noise_inject_node.py
│   ├── image_sharpen_blur_node.py
│   └── image_vignette_glow_node.py
│
├── Latent/                    ← TensorVizion/Latent category (5 nodes)
│   ├── latent_blend_node.py
│   ├── latent_channel_mixer_node.py
│   ├── latent_interpolate_node.py
│   ├── latent_noise_inject_node.py
│   └── latent_visualizer_node.py
│
├── Model Nodes/                ← TensorVizion/Model Utilities category (6 working + 6 draft)
│   ├── clip_text_compare_node.py     ✅ working
│   ├── clip_text_weight_node.py      ✅ working
│   ├── lora_info_node.py             ✅ working
│   ├── lora_stack_node.py            ✅ working
│   ├── model_block_freeze_node.py    ✅ working
│   ├── model_merge_weighted_node.py  ✅ working
│   ├── SDXL_Loader.py                ⚠️ draft, not registered
│   ├── batch_folder_loader.py        ⚠️ draft, not registered
│   ├── dual_model_merger.py          ⚠️ draft, not registered
│   ├── model_info_inspector.py       ⚠️ draft, not registered
│   ├── quick_lora_stacker.py         ⚠️ draft, not registered
│   └── smart_unloader.py             ⚠️ draft, not registered
│
└── Configs/                   ← JSON schema files for select nodes
    ├── audio_beat_detect_node.json
    ├── audio_normalize_node.json
    ├── audio_spectrogram_node.json
    ├── audio_waveform_node.json
    ├── clip_text_compare_node.json
    ├── clip_text_weight_node.json
    ├── lora_info_node.json
    ├── lora_stack_node.json
    ├── model_block_freeze_node.json
    └── model_merge_weighted_node.json
```

---

## Requirements

| Dependency | Notes |
|------------|-------|
| **ComfyUI** | Any recent version |
| **Python 3.9+** | Included with ComfyUI |
| **PyTorch** | Included with ComfyUI |
| **NumPy** | Included with ComfyUI |
| **Pillow** | Included with ComfyUI (used by spectrogram bilinear resize) |

No additional `pip install` steps required for any of the 30 working nodes.
All audio processing (FFT, phase vocoder, reverb, beat detection) and image
processing is implemented in pure NumPy/PyTorch.

---

## Troubleshooting

**Nodes do not appear after install**
Restart ComfyUI completely. Check the terminal for `[OmniNodes]` log lines —
`✅ Loaded` means the file registered, `⚠️ No NODE_CLASS_MAPPINGS` means the
file was found but skipped (this is expected and harmless for the 6 draft
files in `Model Nodes/`, see [Known Issues](#known-issues--draft-nodes)), and
`❌ Error importing` means a real failure with a traceback printed below it.

**Import error on a specific node**
Read the traceback in the ComfyUI terminal. Most failures are missing
ComfyUI internals (e.g. `folder_paths`, `comfy.sd`) which usually means either
the pack isn't inside `ComfyUI/custom_nodes/`, or — for the 6 draft files
specifically — the import was never added in the first place.

**LoRA Stack / LoRA Info Inspector say `LoRA not found`**
LoRA filenames come from ComfyUI's `folder_paths` registry. Make sure your
LoRAs are in the folder ComfyUI expects (usually `ComfyUI/models/loras/`).

**Audio nodes produce silence**
Check that your AUDIO source node outputs a `waveform` key in its dict, or a
raw tensor — OmniNodes accepts both formats. If `peak_db` from Audio
Normalize returns `-inf`, the incoming audio is all zeros.

**A "Model Loaders" node I expected to see isn't in the menu**
That's one of the 6 draft files in `Model Nodes/` — see
[Known Issues](#known-issues--draft-nodes). It exists in the folder but isn't
wired into the loader yet.

---

## Where to Go Next

The pack has grown from 16 → 30 working nodes across Audio, Image, and Model
Utilities, but a few gaps stand out as the clearest next additions:

**1. Finish the 6 draft Model Loader nodes first.** This is lower effort than
new development — the logic is already sketched out, they just need imports,
a `NODE_CLASS_MAPPINGS` block, and (for `dual_model_merger.py`) a corrected
merge call. `SDXL_Loader`, `Smart Unloader`, and `Quick LoRA Stacker` in
particular cover real workflow gaps (one-node checkpoint+VAE loading, VRAM
cleanup between batch runs, and a lighter alternative to the full 5-slot LoRA
Stack) that the currently-working nodes don't address.

**2. Image category is the newest and thinnest (6 nodes vs. Audio's 13).**
Natural next additions, mirroring what Audio already has: an **Image
Load/Save** pair for explicit file I/O outside the standard ComfyUI
Load/Save Image nodes (useful if you want OmniNodes-native metadata
handling), an **Image Resize/Crop** node with the same aspect-aware logic as
Audio Resample, and an **Image Channel Mixer** to complete the Image ↔
Latent parity you've already established (Blend, Noise Inject, and Mask
Composite all have Latent counterparts now — Channel Mixer doesn't yet).

**3. A Latent ↔ Image bridge utility.** You have Latent Visualizer (latent →
preview image) but nothing generalized for round-tripping metadata or stats
between the two spaces — e.g. a node that overlays Latent Visualizer stats as
text onto its own output image, useful for debugging grids without a second
node.

**4. Batch/orchestration nodes**, since your production workflow (per your
past sessions) already leans on Prompt Queue, Seed Bank, and Contact Sheet
concepts — a **LoRA Recipe Save/Load** pair (serializing a full LoRA Stack
configuration to JSON and back) would pair naturally with Model Recipe Writer
and let you version-control "known good" merge/LoRA combinations the same way
you already do for prompts.

**5. Video/animation support** is conspicuously absent given Latent
Interpolate already generates batched morph latents — a simple **Batch to
GIF/MP4** export node would close the loop from "generate an interpolation
batch" to "get a shareable file" without leaving the pack.

If you tell me which of these you want to tackle first, I can help scaffold
the actual node file(s) in the same style/conventions as the rest of the pack.

---

## License

MIT — see `LICENSE` file.

---

*OmniNodes by TensorVizion · github.com/TensorVizion/OmniNodes*
