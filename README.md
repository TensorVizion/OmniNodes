# OmniNodes — ComfyUI Custom Node Pack

> **By TensorVizion** · 16 nodes · 3 categories · Zero heavy dependencies

A production-grade ComfyUI custom node pack covering audio processing, latent space
manipulation, and model utilities — built entirely on PyTorch and NumPy with no
external audio libraries required.

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

Restart ComfyUI after installing. All 16 nodes will appear automatically under
the **TensorVizion/** category group in the node search menu.

---

## Node Reference

### 🎵 Audio Nodes — `TensorVizion/Audio`

| Node | Inputs | Outputs |
|------|--------|---------|
| **Audio Beat Detect 🥁** | `audio_samples`, `sample_rate`, `sensitivity`, `min_bpm`, `max_bpm` | `beat_times` STRING · `beat_count` INT · `bpm` FLOAT · `summary` STRING |
| **Audio Mixer 🎚️** | `master_gain_db` · 4× tracks each with `audio` (optional), `gain_db`, `pan`, `mute` | `audio` AUDIO · `summary` STRING |
| **Audio Normalize 🔊** | `audio_samples`, `target_db`, `normalize_mode`, `remove_dc_offset`, `soft_clip` | `audio` AUDIO · `peak_db` FLOAT · `rms_db` FLOAT · `summary` STRING |
| **Audio Pitch Shift 🎼** | `audio_samples`, `semitones`, `fft_size`, `hop_length`, `preserve_formants` | `audio` AUDIO · `summary` STRING |
| **Audio Reverb 🏛️** | `audio_samples`, `sample_rate`, `reverb_type`, `room_size`, `damping`, `wet_level`, `dry_level`, `pre_delay_ms`, `stereo_width` | `audio` AUDIO · `summary` STRING |
| **Audio Spectrogram 🎛️** | `audio_samples`, `sample_rate`, `width`, `height`, `fft_size`, `hop_length`, `spectrogram_type`, `colormap` | `spectrogram_image` IMAGE |
| **Audio Waveform 🎵** | `audio_samples`, `width`, `height`, `line_color_r/g/b`, `bg_color_r/g/b`, `line_thickness` | `waveform_image` IMAGE |

#### Audio node details

**Audio Beat Detect** — Energy-based onset detection. Returns timestamps as a
comma-separated string, total beat count, and estimated BPM clamped to
`[min_bpm, max_bpm]`. `sensitivity` controls how much louder a frame must be
relative to its local window to count as a beat (default 1.5×).

**Audio Mixer** — 4-channel stereo mixer. Each track has independent `gain_db`
(−60 to +12 dB), constant-power stereo `pan` (−1 left → +1 right), and a `mute`
toggle. Tracks are optional — leave sockets disconnected to leave slots empty.
Output is protected by a tanh soft-clipper.

**Audio Normalize** — Peak or RMS normalisation to a target dBFS level. Optional
DC offset removal (mean subtraction) runs before level detection. `soft_clip`
applies a tanh limiter at the output stage.

**Audio Pitch Shift** — Phase-vocoder pitch shift with ±24 semitone range (2
octaves). Implemented in pure NumPy — no librosa or soundfile required. Processes
each channel independently; mono inputs are handled transparently.

**Audio Reverb** — Two modes: `algorithmic` (Schroeder comb + allpass filter
network) and `convolution` (synthetic decaying impulse response via FFT
convolution). `room_size` scales delay times and IR length. `stereo_width`
widens the mid/side image of the wet signal.

**Audio Spectrogram** — STFT-based spectrogram rendered as a ComfyUI IMAGE.
`spectrogram_type`: `linear` uses raw magnitude; `log_power` applies log1p for
perceptual scaling. Four colormaps: `cyan_dark`, `viridis`, `hot`, `grayscale`.
Frequency axis is flipped so low frequencies sit at the bottom.

**Audio Waveform** — Renders a waveform visualisation as a ComfyUI IMAGE. Stereo
inputs are collapsed to mono by averaging. RGB colour is set per-channel for both
the waveform line and the background.

---

### 🔧 Model Utilities — `TensorVizion/Model Utilities`

| Node | Inputs | Outputs |
|------|--------|---------|
| **CLIP Text Compare 🔍** | `clip`, `prompt_a`, `prompt_b` | `conditioning_a` · `conditioning_b` · `similarity` FLOAT · `similarity_label` STRING |
| **CLIP Text Weight ⚖️** | `clip`, `prompt`, `weight` · optional: `conditioning_in`, `blend_ratio` | `conditioning` CONDITIONING |
| **LoRA Info Inspector 🔬** | `lora_name` | `summary` STRING · `key_count` INT · `rank` INT · `alpha` FLOAT · `target_modules` STRING |
| **LoRA Stack 🗂️** | `model`, `clip` · 5× `lora_N`, `model_weight_N`, `clip_weight_N` | `model` MODEL · `clip` CLIP · `stack_summary` STRING |
| **Model Block Freeze 🧊** | `model`, `freeze_input/middle/output`, block range ints, `freeze_strength` | `model` MODEL · `freeze_summary` STRING |
| **Model Merge Weighted 🔀** | `model_a`, `model_b`, `ratio`, `strategy` | `merged_model` MODEL |

#### Model utility details

**CLIP Text Compare** — Encodes two prompts and computes cosine similarity on the
pooled CLIP embeddings. Returns both conditionings (ready to use downstream) plus
a `similarity` float in [0, 1] and a human-readable label. Useful for prompt
A/B testing and measuring semantic distance between concepts.

**CLIP Text Weight** — Encodes a prompt and multiplies the conditioning tensor by
`weight`, providing a numeric slider alternative to `(prompt:1.3)` bracket syntax.
When `conditioning_in` is connected, blends the new conditioning with the incoming
one at `blend_ratio` (0 = all incoming, 1 = all new). Supports negative weights
for concept negation.

**LoRA Info Inspector** — Reads a LoRA file and returns its structural metadata
without applying it to any model: key count, detected rank (from `lora_down`
tensor shape), alpha value, estimated parameter count, and a list of unique target
module prefixes. Use this to audit unfamiliar LoRAs before loading them.

**LoRA Stack** — Applies up to 5 LoRAs in sequence with individual `model_weight`
and `clip_weight` per slot. Slots set to `None` are skipped. `stack_summary`
lists every applied LoRA with its weights. Replaces chaining multiple Load LoRA
nodes.

**Model Block Freeze** — Applies zero-weight patches to selected input, middle,
or output block ranges of a diffusion UNet. `freeze_strength` at 0.0 fully
neutralises the selected blocks; at 1.0 leaves them unchanged (effectively a
no-op). Useful for style ablation and partial-inference experiments.

**Model Merge Weighted** — Three merge strategies:
- `weighted_sum` — classic linear blend: `out = A·(1−r) + B·r`
- `sigmoid_blend` — smooth S-curve applied across layer depth; layers near the
  `ratio` boundary blend softly while distant layers snap to A or B
- `layer_select` — encoder blocks from A, decoder blocks from B, split at `ratio`

---

### 🌀 Latent — `TensorVizion/Latent`

| Node | Inputs | Outputs |
|------|--------|---------|
| **Latent Blend 🌀** | `latent_a`, `latent_b`, `blend_mode`, `ratio`, `strength`, `normalize_output` · optional: `mask` | `latent` LATENT · `blend_info` STRING |
| **Latent Mask 🎭** | `latent`, `mask_shape`, `blend_mode`, `x`, `y`, `width`, `height`, `feather`, `clamp_value`, `invert_mask`, `seed` · optional: `mask` | `latent` LATENT · `mask_out` MASK |
| **Latent Noise Inject 🌊** | `latent`, `noise_type`, `blend_mode`, `strength`, `seed`, `channel_mask` | `latent` LATENT |

#### Latent node details

**Latent Blend** — Blends two latent tensors using one of nine modes:
`lerp`, `add`, `subtract`, `multiply`, `screen`, `overlay`, `difference`,
`hardlight`, `spatial_mask`. Photoshop-style modes (`screen`, `overlay`,
`hardlight`) normalise to [0, 1] per-channel before blending, then denormalise
back. `spatial_mask` mode uses a connected MASK to blend spatially — A outside
the mask, B inside. If the two latents differ in spatial size, B is resized to
match A automatically.

**Latent Mask** — Applies a geometric or external mask to a latent tensor.
Built-in shapes: `rectangle`, `ellipse`, `gradient_h`, `gradient_v`. Set
`mask_shape` to `external` and connect a MASK socket to use your own mask.
`feather` softens shape edges via distance-based falloff. Four blend modes for
the masked region: `zero` (erase), `noise` (fill with seeded gaussian), `clamp`
(constrain to ±`clamp_value`), `invert` (negate). `invert_mask` flips which
region is affected. Outputs both the modified latent and the generated mask.

**Latent Noise Inject** — Adds structured noise directly to a latent tensor.
Noise types: `gaussian`, `uniform`, `perlin` (4-octave approximate via bilinear
upsampling), `salt_pepper` (sparse ±1 spike noise). Blend modes: `add`,
`multiply`, `lerp`. `channel_mask` accepts `all` or a comma-separated list of
channel indices (e.g. `0,1`) to restrict injection to specific latent channels.
`seed` makes noise fully reproducible.

---

## Folder Structure

```
OmniNodes/
├── __init__.py               ← loads all nodes at ComfyUI startup
├── README.md
├── pyproject.toml
│
├── Audio Nodes/              ← TensorVizion/Audio category (7 nodes)
│   ├── audio_beat_detect_node.py
│   ├── audio_mixer_node.py
│   ├── audio_normalize_node.py
│   ├── audio_pitch_shift_node.py
│   ├── audio_reverb_node.py
│   ├── audio_spectrogram_node.py
│   └── audio_waveform_node.py
│
├── Model Utilities/          ← TensorVizion/Model Utilities category (6 nodes)
│   ├── clip_text_compare_node.py
│   ├── clip_text_weight_node.py
│   ├── lora_info_node.py
│   ├── lora_stack_node.py
│   ├── model_block_freeze_node.py
│   └── model_merge_weighted_node.py
│
├── Latent/                   ← TensorVizion/Latent category (3 nodes)
│   ├── latent_blend_node.py
│   ├── latent_mask_node.py
│   └── latent_noise_inject_node.py
│
└── Configs/                  ← JSON schema files for each node
    ├── audio_beat_detect_node.json
    ├── audio_mixer_node.json
    └── ... (one per node)
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

No additional `pip install` steps required. All audio processing (FFT, phase
vocoder, reverb, beat detection) is implemented in pure NumPy.

---

## Troubleshooting

**Nodes do not appear after install**
Restart ComfyUI completely. Check the terminal for `[OmniNodes]` log lines — a
`✓` means the node loaded, `✗` means it failed with a traceback printed below.

**`[OmniNodes] ✗  Folder missing: 'Latent/'`**
The `Latent/` sub-folder must exist at the same level as `__init__.py`. If you
installed via zip, make sure you extracted the full folder structure and didn't
flatten the files.

**Import error on a specific node**
Read the traceback in the ComfyUI terminal. Most failures are missing ComfyUI
internals (e.g. `folder_paths`) which means the node file ended up somewhere
ComfyUI can't inject its environment from — confirm the pack is inside
`ComfyUI/custom_nodes/`.

**LoRA Stack says `LoRA not found`**
The LoRA filenames in the dropdown come from ComfyUI's `folder_paths` registry.
Make sure your LoRAs are in the folder ComfyUI expects (usually
`ComfyUI/models/loras/`).

**Audio nodes produce silence**
Check that your AUDIO source node outputs a `waveform` key in its dict, or a raw
tensor. OmniNodes accepts both formats. If `peak_db` from Audio Normalize returns
`-inf`, the incoming audio is all zeros.

---

## License

MIT — see `LICENSE` file.

---

*OmniNodes by TensorVizion · github.com/TensorVizion/OmniNodes*
