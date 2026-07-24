# OmniNodes — ComfyUI Custom Node Pack

> **By TensorVizion** · 50 working nodes · 6 categories · Zero heavy dependencies
> All nodes verified working — see [Changelog](#changelog) for what was recently fixed and added.

A production-grade ComfyUI custom node pack covering audio processing, image
post-processing, latent space manipulation, model utilities, prompt/wildcard
tooling, and workflow control — built almost entirely on PyTorch, NumPy, and
the Python standard library, with no external audio/image libraries required.

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
`NODE_CLASS_MAPPINGS` dict or it will be silently skipped.

Working nodes appear under the **TensorVizion/** category group in the node
search menu, split into six sub-groups: `Audio`, `Image`, `Latent`,
`Model Utilities`, `Prompt`, and `Workflow`.

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

### 🖼️ Image Nodes — `TensorVizion/Image` (10 nodes)

| Node | Inputs | Outputs |
|------|--------|---------|
| **Image Load 📂** | `image_file` (dropdown from `input/`) · optional `file_path_override` | `image` IMAGE · `mask_out` MASK · `summary` STRING |
| **Image Save 💾** | `images`, `filename_prefix`, `format`, `quality`, `embed_workflow` | `saved_paths` STRING |
| **Image Resize & Crop 📐** | `image`, `resize_mode`, `target_width`, `target_height`, `interpolation`, `crop_x`, `crop_y`, `pad_color` | `image` IMAGE · `output_width` INT · `output_height` INT · `summary` STRING |
| **Image Channel Mixer 🎚️** | `image`, `red_mix`, `green_mix`, `blue_mix`, `red/green/blue_offset`, `clamp_output` | `image` IMAGE · `mix_info` STRING |
| **Image Blend 🖌️** | `image_a`, `image_b`, `blend_mode`, `ratio`, `strength`, `clamp_output` · optional `mask` | `image` IMAGE · `blend_info` STRING |
| **Image Color Grade 🎨** | `image`, `exposure`, `contrast`, `saturation`, `gamma`, `lift`, `gain`, `temperature`, `tint` | `image` IMAGE · `grade_info` STRING |
| **Image Mask Composite 🖼️** | `image`, `mask_shape`, `effect`, `x`, `y`, `width`, `height`, `feather`, `invert_mask`, `darken_amount`, `brighten_amount`, `blur_radius`, `color_r/g/b` · optional `mask` | `image` IMAGE · `mask_out` MASK |
| **Image Noise Inject 🎞️** | `image`, `noise_type`, `blend_mode`, `strength`, `seed`, `monochrome`, `clamp_output` | `image` IMAGE |
| **Image Sharpen & Blur 🔎** | `image`, `mode`, `radius`, `strength` | `image` IMAGE |
| **Image Vignette & Glow ✨** | `image`, `vignette_amount`, `vignette_radius`, `glow_threshold`, `glow_strength`, `glow_radius` | `image` IMAGE |

#### Image node details

This category is the pixel-space counterpart to the Latent category below —
several nodes are direct analogues (Image Blend ↔ Latent Blend, Image Noise
Inject ↔ Latent Noise Inject, Image Channel Mixer ↔ Latent Channel Mixer).
Image Load/Save/Resize round out the category with the same "entry point /
exit point / dimension change" roles that Audio Load/Save/Resample fill for
the Audio category.

**Image Load** — Entry point for image workflows built entirely within
TensorVizion/Image, as an alternative to ComfyUI's core Load Image node.
Reads PNG/JPEG/WEBP/BMP from ComfyUI's `input/` folder (or an absolute path
override), auto-corrects EXIF rotation (common with phone photos), and
splits any alpha channel out as a MASK — if the source has no alpha, returns
a full-opacity mask so downstream mask-consuming nodes always get a valid
input rather than `None`.

**Image Save** — Exit point for image workflows. Writes PNG (lossless, with
optional embedded workflow metadata via PNG text chunks — the same
mechanism ComfyUI's own Save Image node uses, so files can be dragged back
into ComfyUI to restore the workflow), JPEG, or WEBP (both lossy, smaller,
no metadata embedding since those formats don't carry PNG text chunks).
Follows ComfyUI's standard `prefix_00001_.png` numbering so repeat runs
never overwrite previous saves.

**Image Resize & Crop** — Four resize strategies in one node: `stretch`
(exact target size, ignores aspect ratio), `fit` (preserves aspect ratio,
letterboxed/padded to the exact target canvas with `pad_color`), `fill`
(preserves aspect ratio, fills the target completely and center-crops any
overflow), and `crop_only` (no resizing — crops a `crop_width x
crop_height` region at `crop_x, crop_y`). `interpolation`: `bilinear`
(smooth default), `nearest` (blocky — best for pixel art or masks), `area`
(best for heavy downscaling, reduces moiré).

**Image Channel Mixer** — A classic RGB channel mixer, same operation as
Photoshop/GIMP's Channel Mixer tool: each output channel is a weighted sum
of the input R/G/B channels (`red_mix`/`green_mix`/`blue_mix` as
comma-separated `"r,g,b"` weights) plus a per-channel offset. Distinct from
Latent Channel Mixer's per-channel gain/offset-only approach, since IMAGE
tensors have a fixed, meaningful 3-channel structure where cross-channel
mixing does something visually useful — luminance-weighted grayscale
conversion, channel swaps, or a sepia tone are all single-node presets (see
the node's docstring for exact weight values).

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

### 🔧 Model Utilities — `TensorVizion/Model Utilities` (12 nodes)

| Node | Inputs | Outputs |
|------|--------|---------|
| **CLIP Text Compare 🔍** | `clip`, `prompt_a`, `prompt_b` | `conditioning_a` · `conditioning_b` · `similarity` FLOAT · `similarity_label` STRING |
| **CLIP Text Weight ⚖️** | `clip`, `prompt`, `weight` · optional `conditioning_in`, `blend_ratio` | `conditioning` CONDITIONING |
| **LoRA Info Inspector 🔬** | `lora_name` | `summary` STRING · `key_count` INT · `rank` INT · `alpha` FLOAT · `target_modules` STRING |
| **LoRA Stack 🗂️** | `model`, `clip` · 5× `lora_N`, `model_weight_N`, `clip_weight_N` | `model` MODEL · `clip` CLIP · `stack_summary` STRING |
| **Model Block Freeze 🧊** | `model`, `freeze_input/middle/output`, block range ints, `freeze_strength` | `model` MODEL · `freeze_summary` STRING |
| **Model Merge Weighted 🔀** | `model_a`, `model_b`, `ratio`, `strategy` | `merged_model` MODEL |
| **Simple SDXL Loader 📀** | `model_name`, `vae_auto`, `vae_name` | `model` MODEL · `clip` CLIP · `vae` VAE · `summary` STRING |
| **Batch Folder Loader 📂** | `subfolder`, `filter_extensions`, `load_first` | `model` MODEL · `clip` CLIP · `vae` VAE · `found_models` STRING |
| **Dual Model Merger 🔀** | `model_A`, `model_B`, `merge_ratio`, `interpolation` | `model` MODEL · `merge_info` STRING |
| **Model Info Inspector 🔬** | `ckpt_name` | `summary` STRING · `key_count` INT · `architecture` STRING · `precision` STRING |
| **Quick LoRA Stacker ⚡** | `model`, `clip` · 3× `lora_N`, `strength_N` | `model` MODEL · `clip` CLIP · `stack_summary` STRING |
| **Smart Unloader 🧹** | `passthrough` (any type), `unload_all`, `empty_cache`, `run_gc` | `output` (passthrough) · `summary` STRING |

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

**Simple SDXL Loader** — Loads a checkpoint's MODEL/CLIP/VAE in one node. If
the checkpoint has no bundled VAE and `vae_auto` is on, it looks for a
same-named `*.vae.safetensors` file in your `vae/` folder automatically
(e.g. `MyModel.safetensors` → `MyModel.vae.safetensors`). Set `vae_name` to
force a specific external VAE regardless of auto-detection. `summary`
reports which VAE source was actually used, so you're never guessing.

**Batch Folder Loader** — Lists every checkpoint under a chosen subfolder of
ComfyUI's registered `checkpoints/` search path (not an arbitrary filesystem
path — this only sees files ComfyUI itself can already find), filtered by
extension. With `load_first` on, also loads the first match's MODEL/CLIP/VAE
so the node can sit directly in a workflow instead of being pure inspection.
Handy for "load whatever's newest in this folder" setups or auditing what's
actually present in a subfolder.

**Dual Model Merger** — A friendlier-named sibling to Model Merge Weighted
with classic A1111-style method names: `Weighted Sum`, `Add Difference`
(mathematically identical to Weighted Sum for a two-model blend, offered
under this name for people who think of B as a "delta" on A), and `Slerp`
(a sinusoidal easing curve around the midpoint rather than a straight linear
ramp — an approximation of spherical interpolation at the patch-strength
level, since ComfyUI's patch API doesn't expose raw tensors directly; for
literal per-tensor SLERP on latents rather than model weights, see Latent
Interpolate's `slerp` method instead).

**Model Info Inspector** — Reads a checkpoint's raw state dict and reports
key count, a best-guess architecture (FLUX, SD3, SDXL, SD1.x/2.x, or
unrecognized, based on key-name signatures), detected weight precision, and
an estimated parameter count — without running a full
`load_checkpoint_guess_config`, so it works even on checkpoints ComfyUI can't
fully load, and it's faster than a full load since no submodules are built.

**Quick LoRA Stacker** — A lighter 3-slot alternative to LoRA Stack, using
one shared strength value per slot (applied to both model and clip) instead
of independent model/clip weights. Use this for the common case where you
don't need LoRA Stack's full dual-weight control surface.

**Smart Unloader** — Frees VRAM at the exact point it executes in your
graph, then passes its input straight through unchanged so it can sit inline
in a workflow without breaking downstream connections. `unload_all` calls
ComfyUI's full model-unload; `empty_cache` clears the CUDA cache;
`run_gc` runs Python's garbage collector. `summary` reports before/after VRAM
usage when CUDA is available. Accepts and returns **any** ComfyUI type
(MODEL, IMAGE, LATENT, CONDITIONING, etc.) on its passthrough socket.

---

### 🎲 Prompt Nodes — `TensorVizion/Prompt` (5 nodes)

| Node | Inputs | Outputs |
|------|--------|---------|
| **Wildcard Loader 🎲** | `text`, `seed`, `max_recursion` · optional `wildcards_subfolder` | `text` STRING · `resolution_log` STRING · `seed_used` INT |
| **Wildcard List Inspector 📋** | `preview_lines` | `listing` STRING · `file_count` INT |
| **Prompt Combiner ➕** | `separator`, `weight_1..4` · optional `text_1..4` | `combined_text` STRING |
| **Prompt Random Line 🎯** | `text`, `seed` | `chosen_line` STRING · `line_index` INT · `total_lines` INT |
| **Prompt Cleaner 🧹** | `text`, `case_insensitive_dedup`, `lowercase`, `max_tags` | `cleaned_text` STRING · `removed_count` INT |

#### Prompt node details

**Wildcard Loader** — Resolves `__wildcard_name__` tokens against `.txt`
files in a `wildcards/` folder, plus inline `{option_a|option_b|option_c}`
dynamic-prompt syntax, in the same pass. Looks for `<ComfyUI root>/wildcards/`
first (the same shared convention Impact Pack/Dynamic Prompts-style
extensions use, so an existing wildcard collection is picked up
automatically), and falls back to a bundled `wildcards/` folder inside this
pack — auto-created with two starter files (`color.txt`, `quality.txt`) on
first run so the node has something to resolve with zero setup. Lines
inside a wildcard file support an optional `N::` weight prefix (e.g.
`2::crimson`) for weighted random selection; unweighted lines default to
weight 1. Tokens can nest — a line picked from one file can itself contain
another `__token__` or `{a|b}` choice, resolved recursively up to
`max_recursion` levels (guards against circular references like a.txt
containing `__b__` and b.txt containing `__a__`). `seed` makes the whole
expansion reproducible: the same seed always resolves the same prompt to
the same output.

**Wildcard List Inspector** — Companion discovery node for Wildcard Loader.
Scans the same `wildcards/` folder and reports every `.txt` file found —
its wildcard name (what goes inside the `__double_underscores__`), how many
usable lines it has, and a short preview — so you don't need to open a file
browser to remember what's available.

**Prompt Combiner** — Joins up to 4 optional text fragments into one
string. Each connected fragment gets its own `weight_N`; a weight of 1.0
passes through unchanged, anything else is wrapped in `(fragment:weight)`
syntax before joining — the standard emphasis format most SD/SDXL
frontends (and this pack's own CLIP Text Weight node) understand. Empty or
disconnected slots are skipped entirely, so there are never dangling
separators. `separator` defaults to `", "` but accepts anything, including
`\n` for one-fragment-per-line prompts.

**Prompt Random Line** — Splits a pasted multiline `text` on newlines and
returns one line chosen by `seed`. The inline, file-free companion to
Wildcard Loader — useful for a short one-off list of alternatives typed
directly into the node rather than managed as a separate `.txt` file.
Blank lines and `#`-prefixed comment lines are ignored when counting and
selecting.

**Prompt Cleaner** — Normalises a prompt after concatenation or wildcard
resolution has left it messy: splits on commas, strips whitespace, removes
exact duplicate tags (case-insensitive by default, keeping the first
occurrence), collapses internal whitespace runs, and drops empty tags left
behind by stray double-commas. Optional `lowercase` and `max_tags`
(0 = no limit) truncation. `removed_count` reports how many tags were
actually stripped, so you can tell whether cleaning did anything on a
given run.

---

### 🏁 Workflow Nodes — `TensorVizion/Workflow` (5 nodes)

| Node | Inputs | Outputs |
|------|--------|---------|
| **Workflow End 🏁** | `run_label`, `write_log`, `write_done_marker` · optional `input_1..4` (any type) | `summary` STRING |
| **Any Switch 🔀** | `condition` · optional `input_a`, `input_b` (any type) | `output` (any type) · `selected_label` STRING |
| **Batch Counter 🔢** | `counter_id`, `step`, `reset`, `base_seed`, `seed_step` | `count` INT · `derived_seed` INT |
| **Timer Start ⏱️▶️** | `timer_id` · optional `passthrough` (any type) | `output` (passthrough) |
| **Timer Stop ⏱️⏹️** | `timer_id` · optional `passthrough` (any type) | `output` (passthrough) · `elapsed_seconds` FLOAT · `summary` STRING |

#### Workflow node details

**Workflow End** — An explicit terminal node for a workflow. Connect any
final outputs you want to make sure actually execute into the four
optional `input_N` sockets (accepts any ComfyUI type — same `AlwaysEqualProxy`
wildcard-type trick Smart Unloader uses) — none of them need to be used for
anything else afterward, so this is a clean place to let dangling branches
terminate instead of leaving sockets unconnected. With `write_log` on,
appends a timestamped line to `omninodes_run_log.txt` in ComfyUI's
`output/` folder. With `write_done_marker` on, also writes a small
`<run_label>.done` file to `output/` — a simple signal file for external
automation (batch scripts, queue managers) watching the output folder
rather than polling ComfyUI's HTTP API. Registered as `OUTPUT_NODE = True`,
so ComfyUI's UI marks it as a graph endpoint the way Save Image/Save Audio
are marked.

**Any Switch** — Routes `input_a` or `input_b` to `output` based on a
single `condition` boolean, for any ComfyUI type (MODEL, IMAGE, LATENT,
CONDITIONING, STRING, etc.) — one reusable switch instead of needing a
separate type-specific switch node per data type. `selected_label` reports
which slot was actually chosen (`"a"` or `"b"`) as a STRING, handy for
feeding into a summary/log node or a Workflow End `run_label`.

**Batch Counter** — A persistent, file-backed counter (stored under
ComfyUI's `output/` folder as `<counter_id>.counter`) that increments once
per queued run and survives across ComfyUI restarts, unlike an in-memory
counter. `derived_seed` = `base_seed + count * seed_step` — plug straight
into any seed input downstream for a fresh, reproducible seed every queue
without manually incrementing anything. `counter_id` lets multiple
independent counters run side by side. Uses `IS_CHANGED` returning `NaN` to
force a real execution every queue — without it, ComfyUI's result cache
would treat identical widget inputs as identical runs and the counter would
never advance.

**Timer Start / Timer Stop** — A pair for benchmarking a section of a
workflow. Give both nodes the same `timer_id`; Timer Start records a
timestamp when it executes, Timer Stop reads it back and reports
`elapsed_seconds` plus a human-readable `summary`. Connect `passthrough` on
each to whatever output starts/ends the section you're measuring — the
same dependency trick Smart Unloader uses to force correct execution order
in the graph. Multiple independent timers can run at once with different
`timer_id` values. If Timer Stop runs without a matching Timer Start this
queue, `elapsed_seconds` returns `-1` and `summary` says so explicitly
rather than silently reporting `0`. The shared start-time dict lives as an
attribute on `comfy.model_management` (rather than a direct cross-file
import) since each file in this pack is loaded as an independent module by
the auto-discovery loader, with no shared package namespace between them.

---

## Changelog

**This build — 10 new nodes across two new categories: Prompt and Workflow.**

Prompt Nodes (`TensorVizion/Prompt`, 5 nodes) — file-and-inline wildcard
resolution plus general prompt-string utilities:

| Node | Role |
|------|------|
| `Wildcard Loader` | Resolves `__name__` tokens against wildcard `.txt` files (weighted lines, recursive nesting) and inline `{a\|b\|c}` choices in one pass, seeded for reproducibility |
| `Wildcard List Inspector` | Lists every wildcard file available, with line counts and previews |
| `Prompt Combiner` | Joins up to 4 weighted text fragments into one prompt string |
| `Prompt Random Line` | Picks one seeded random line from a pasted multiline list — the file-free companion to Wildcard Loader |
| `Prompt Cleaner` | Strips duplicate/empty comma-separated tags and normalises whitespace |

Workflow Nodes (`TensorVizion/Workflow`, 5 nodes) — graph control and
run-tracking utilities:

| Node | Role |
|------|------|
| `Workflow End` | Explicit terminal node — any-type sink, optional run log + done-marker file for external automation |
| `Any Switch` | Boolean-gated router for any ComfyUI type, replacing a per-type switch node |
| `Batch Counter` | Persistent, file-backed counter + derived seed, increments once per queued run |
| `Timer Start` / `Timer Stop` | Paired benchmarking nodes reporting elapsed time across a section of a workflow |

All 10 were verified with real functional tests against each function
directly (not just import checks): Wildcard Loader's seeded reproducibility
and weighted-line distribution over 500 seeds, Batch Counter's persistence
and increment behavior across repeated calls (plus its `IS_CHANGED` cache
bypass), Timer Start/Stop's shared-state handoff across two separately
loaded files, Workflow End's actual log-file and done-marker writes, and
straightforward input/output checks for the remaining Prompt nodes. The
full pack (all 50 nodes together) was also re-run through the real
`__init__.py` auto-discovery loader against mocked ComfyUI modules to
confirm zero name collisions and zero import-order issues with the new
files.

**Pack total: 50 working nodes** (previously 40).

---

**Earlier build — all 6 draft nodes fixed and now production-ready:**

| Node | What was wrong | What was fixed |
|------|------------------|------------------|
| `SDXL_Loader.py` | No `NODE_CLASS_MAPPINGS`; missing imports; used a non-existent `folder_paths.exists_annotated_file` API | Added `folder_paths`/`comfy.sd`/`comfy.utils` imports, registered the node, replaced the fake API with `comfy.sd.VAE(sd=...)` construction from a loaded state dict, added a `summary` output reporting which VAE was actually used |
| `batch_folder_loader.py` | No `NODE_CLASS_MAPPINGS`; missing imports; used a raw filesystem path instead of ComfyUI's model registry | Rebuilt around `folder_paths.get_filename_list("checkpoints")`, scoped to subfolders of ComfyUI's actual registered search paths, added extension filtering and an optional load-first-match behavior |
| `dual_model_merger.py` | No `NODE_CLASS_MAPPINGS`; called `comfy.model_merging.merge_models(interpolation_method=...)`, an API that doesn't exist in ComfyUI | Rebuilt on the real `model.clone()` + `get_key_patches()` + `add_patches()` pattern (same one `Model Merge Weighted` uses), added a sinusoidal easing curve for the `Slerp` option |
| `model_info_inspector.py` | No `NODE_CLASS_MAPPINGS`; missing imports; assumed LoRA-style training metadata (`ss_metadata`) that generic checkpoints don't have | Rebuilt on `comfy.utils.load_torch_file` (the same pattern `LoRA Info Inspector` already uses successfully), added key-signature-based architecture guessing (FLUX/SD3/SDXL/SD1.x) and precision detection |
| `quick_lora_stacker.py` | No `NODE_CLASS_MAPPINGS`; missing imports; passed a raw path string to `comfy.sd.load_lora_for_models` instead of a loaded state dict | Rebuilt on the same `comfy.utils.load_torch_file` + `comfy.sd.load_lora_for_models` pattern as the working `LoRA Stack` node |
| `smart_unloader.py` | No `NODE_CLASS_MAPPINGS`; missing imports; returned `(None, None, None)` for MODEL/VAE/CLIP, breaking any downstream node | Rebuilt as a passthrough node using an `AlwaysEqualProxy` wildcard type (accepts/returns any ComfyUI type), calls real `comfy.model_management.unload_all_models()` / `soft_empty_cache()`, reports before/after VRAM usage |

All 6 fixes were verified three ways: static structure check (imports,
`NODE_CLASS_MAPPINGS`, `FUNCTION`/`RETURN_TYPES` consistency), a real import
against mocked ComfyUI modules, and actual execution of each node's function
against realistic mock inputs.

**Model Utilities is now 12/12 working nodes** (previously 6/12).

**Also new this build — 4 new Image nodes**, closing most of the gap between
Image (6 nodes) and Audio (13 nodes):

| Node | Role |
|------|------|
| `Image Load` | Entry point for Image-category workflows — reads PNG/JPEG/WEBP/BMP, auto-corrects EXIF rotation, splits alpha into a MASK |
| `Image Save` | Exit point — writes PNG (with optional embedded workflow metadata, matching ComfyUI's own Save Image convention)/JPEG/WEBP |
| `Image Resize & Crop` | Four resize strategies (`stretch`/`fit`/`fill`/`crop_only`) plus interpolation choice — the Image-space counterpart to Audio Resample |
| `Image Channel Mixer` | Classic RGB channel mixer (weighted cross-channel mixing + offset) — completes the Image ↔ Latent parity alongside Blend, Noise Inject, and Mask Composite |

All 4 were verified with real functional tests: actual PNG files written to
disk and re-read to confirm embedded metadata, alpha-channel extraction
checked against known pixel values, all four resize modes checked for
correct output dimensions, and channel-mixer math checked against known
transformations (R/B channel swap, luminance-weighted grayscale).

**Image category is now 10/10 working nodes** (previously 6).

**Pack total: 40 working nodes** (previously 30, before this session's
fixes and additions; 16 in the original release).

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
├── Image Nodes/               ← TensorVizion/Image category (10 nodes)
│   ├── image_load_node.py
│   ├── image_save_node.py
│   ├── image_resize_crop_node.py
│   ├── image_channel_mixer_node.py
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
├── Model Nodes/                ← TensorVizion/Model Utilities category (12 nodes, all working)
│   ├── clip_text_compare_node.py
│   ├── clip_text_weight_node.py
│   ├── lora_info_node.py
│   ├── lora_stack_node.py
│   ├── model_block_freeze_node.py
│   ├── model_merge_weighted_node.py
│   ├── SDXL_Loader.py
│   ├── batch_folder_loader.py
│   ├── dual_model_merger.py
│   ├── model_info_inspector.py
│   ├── quick_lora_stacker.py
│   └── smart_unloader.py
│
├── Prompt Nodes/               ← TensorVizion/Prompt category (5 nodes)
│   ├── wildcard_loader_node.py
│   ├── wildcard_list_inspector_node.py
│   ├── prompt_combiner_node.py
│   ├── prompt_random_line_node.py
│   └── prompt_cleaner_node.py
│
├── Workflow Nodes/             ← TensorVizion/Workflow category (5 nodes)
│   ├── workflow_end_node.py
│   ├── any_switch_node.py
│   ├── batch_counter_node.py
│   ├── timer_start_node.py
│   └── timer_stop_node.py
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

No additional `pip install` steps required for any of the 50 working nodes.
All audio and image processing (FFT, phase vocoder, reverb, beat detection,
color grading, blending, channel mixing) is implemented in pure
NumPy/PyTorch/Pillow. The Model Utilities nodes use ComfyUI's own
`folder_paths`, `comfy.sd`, `comfy.utils`, and `comfy.model_management`
modules, which are already part of any ComfyUI install. The Prompt and
Workflow nodes use only the Python standard library plus `folder_paths` and
`comfy.model_management` — no extra installs there either.

---

## Troubleshooting

**Nodes do not appear after install**
Restart ComfyUI completely. Check the terminal for `[OmniNodes]` log lines —
`✅ Loaded` means the file registered, `⚠️ No NODE_CLASS_MAPPINGS` means the
file was found but skipped, and `❌ Error importing` means a real failure
with a traceback printed below it. All 50 nodes in this build should show
`✅ Loaded`; if any don't, it's a real environment issue (see below), not an
expected gap in the pack.

**Import error on a specific node**
Read the traceback in the ComfyUI terminal. Failures here are almost always
missing ComfyUI internals (e.g. `folder_paths`, `comfy.sd`) not being on the
Python path, which usually means the pack isn't actually inside
`ComfyUI/custom_nodes/` — check the install location first.

**LoRA Stack / Quick LoRA Stacker / LoRA Info Inspector say `LoRA not found`**
LoRA filenames come from ComfyUI's `folder_paths` registry. Make sure your
LoRAs are in the folder ComfyUI expects (usually `ComfyUI/models/loras/`).

**Simple SDXL Loader / Batch Folder Loader can't find a checkpoint**
Same as above — both nodes only see files under ComfyUI's registered
`checkpoints/` search paths, not arbitrary filesystem locations. `Batch
Folder Loader`'s `subfolder` field is relative to that registered root, not
an absolute path.

**Smart Unloader doesn't seem to free any VRAM**
Check the `summary` output string — if CUDA isn't available (CPU-only setup,
or torch built without CUDA), it will report "CUDA not available" and skip
VRAM accounting, since there's nothing to measure. The unload/GC calls still
run either way.

**Audio nodes produce silence**
Check that your AUDIO source node outputs a `waveform` key in its dict, or a
raw tensor — OmniNodes accepts both formats. If `peak_db` from Audio
Normalize returns `-inf`, the incoming audio is all zeros.

---

## License

MIT — see `LICENSE` file.

---

*OmniNodes by TensorVizion · github.com/TensorVizion/OmniNodes*
