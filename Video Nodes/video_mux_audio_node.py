"""
TensorVizion ComfyUI Nodes
video_mux_audio_node.py — Combines an IMAGE batch and an AUDIO track into
a single video file with sound. Video Save closes the loop from "batch in
memory" to "video file", but writes silent video only; every node in the
Audio category (Stem Splitter, Sidechain Duck, Transient Shaper, Beat
Detect, etc.) has nowhere to end up in a final deliverable until this
node exists. Encodes silent video via imageio (same convention as Video
Save), writes the audio to a temp WAV, then muxes both into the final
container with ffmpeg. Falls back to a silent video + a separate .wav
file if ffmpeg isn't available on PATH, rather than failing the queue.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import torch


class VideoMuxAudioNode:
    """
    `images`: the video frames (ComfyUI's IMAGE-batch convention).
    `audio_samples`: an AUDIO input, e.g. from Audio Mixer, Audio
    Sidechain Duck, or Audio Transient Shaper further up the chain.
    `fps` sets playback rate for the video stream; the audio is muxed at
    its own native sample rate (no resampling needed for muxing).

    If the audio is longer or shorter than the video, `audio_fit`
    controls the behavior:
      trim_to_video  — cuts audio to match video duration
      pad_video       — holds the last video frame to match audio duration
      as_is           — muxes both at native length (most players will
                        stop at the shorter of the two streams)

    Requires `imageio` for the silent video pass and `ffmpeg` on PATH for
    the actual mux; if ffmpeg isn't found, saves the silent video and a
    companion .wav file side-by-side instead of failing outright.
    """

    CATEGORY = "TensorVizion/Video"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio_samples": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "filename": ("STRING", {"default": "tv_video_audio"}),
                "output_dir": ("STRING", {"default": "output/tensorvizion/video"}),
                "format": (["mp4", "webm"], {"default": "mp4"}),
                "audio_fit": (["trim_to_video", "pad_video", "as_is"], {"default": "trim_to_video"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_path", "summary")
    FUNCTION = "run"

    @staticmethod
    def _tensor_to_uint8(img_tensor):
        arr = img_tensor.cpu().numpy()
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    def _write_wav(self, audio_samples, wav_path):
        import wave

        if isinstance(audio_samples, dict):
            waveform = audio_samples.get("waveform", audio_samples.get("samples"))
            sample_rate = int(audio_samples.get("sample_rate", 44100))
        else:
            waveform = audio_samples
            sample_rate = 44100

        if isinstance(waveform, torch.Tensor):
            samples = waveform.squeeze().float().cpu().numpy()
        else:
            samples = np.asarray(waveform, dtype=np.float32).squeeze()

        if samples.ndim == 1:
            samples = samples[np.newaxis, :]  # (1, N) mono
        n_channels = samples.shape[0]

        interleaved = samples.T  # (N, channels)
        clipped = np.clip(interleaved, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())

        duration = samples.shape[1] / float(sample_rate)
        return duration, sample_rate

    def run(self, images, audio_samples, fps, filename, output_dir, format, audio_fit):
        try:
            import imageio.v2 as imageio
        except ImportError:
            try:
                import imageio
            except ImportError:
                msg = (
                    "[TensorVizion] 'imageio' is not installed. Install it with "
                    "'pip install imageio imageio-ffmpeg' to enable Video Mux Audio."
                )
                return ("", msg)

        os.makedirs(output_dir, exist_ok=True)

        ext = format
        counter = 0
        final_path = os.path.join(output_dir, f"{filename}.{ext}")
        while os.path.exists(final_path):
            counter += 1
            final_path = os.path.join(output_dir, f"{filename}_{counter:03d}.{ext}")

        n_frames = images.shape[0]
        video_duration = n_frames / fps

        tmp_dir = tempfile.mkdtemp(prefix="tv_mux_")
        silent_path = os.path.join(tmp_dir, f"silent.{ext}")
        wav_path = os.path.join(tmp_dir, "audio.wav")

        try:
            audio_duration, sample_rate = self._write_wav(audio_samples, wav_path)

            frames_to_write = list(range(n_frames))
            if audio_fit == "pad_video" and audio_duration > video_duration:
                extra_frames = int((audio_duration - video_duration) * fps)
                frames_to_write = frames_to_write + [n_frames - 1] * extra_frames

            frames = [self._tensor_to_uint8(images[min(i, n_frames - 1)]) for i in frames_to_write]

            writer_kwargs = {"fps": fps, "quality": 8}
            if format == "mp4":
                writer_kwargs["codec"] = "libx264"
            elif format == "webm":
                writer_kwargs["codec"] = "libvpx-vp9"
            imageio.mimsave(silent_path, frames, format=format, **writer_kwargs)

        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return ("", f"[TensorVizion] Failed to prepare video/audio: {e}")

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            # Fall back: keep the silent video + a companion wav rather than failing
            fallback_video = os.path.join(output_dir, f"{filename}_silent.{ext}")
            fallback_wav = os.path.join(output_dir, f"{filename}.wav")
            shutil.copy(silent_path, fallback_video)
            shutil.copy(wav_path, fallback_wav)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            msg = (
                "[TensorVizion] ffmpeg not found on PATH — could not mux audio+video "
                "into one file. Saved silent video and a companion .wav separately:\n"
                f"  video: {fallback_video}\n"
                f"  audio: {fallback_wav}\n"
                "Install ffmpeg and it will be muxed into a single file automatically next run."
            )
            return (fallback_video, msg)

        trim_args = []
        if audio_fit == "trim_to_video":
            trim_args = ["-t", f"{video_duration:.6f}"]

        cmd = [
            ffmpeg_bin, "-y",
            "-i", silent_path,
            "-i", wav_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest" if audio_fit != "pad_video" else "-fflags", "+genpts",
        ] if audio_fit == "pad_video" else [
            ffmpeg_bin, "-y",
            "-i", silent_path,
            "-i", wav_path,
            "-c:v", "copy",
            "-c:a", "aac",
        ] + trim_args

        cmd = cmd + [final_path]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return ("", f"[TensorVizion] ffmpeg mux failed:\n{result.stderr[-800:]}")
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return ("", f"[TensorVizion] ffmpeg mux failed: {e}")

        shutil.rmtree(tmp_dir, ignore_errors=True)

        summary = (
            f"Format:         {format}\n"
            f"Video frames:   {n_frames} @ {fps}fps ({video_duration:.2f}s)\n"
            f"Audio duration: {audio_duration:.2f}s @ {sample_rate}Hz\n"
            f"Audio fit mode: {audio_fit}\n"
            f"Saved to:       {final_path}"
        )

        return (final_path, summary)


NODE_CLASS_MAPPINGS = {
    "VideoMuxAudioNode": VideoMuxAudioNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoMuxAudioNode": "Video Mux Audio 🔊",
}
