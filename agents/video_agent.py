"""
agents/video_agent.py — Agent 6: Video assembly via FFmpeg subprocess only.

ROOT CAUSE OF TERMINAL CRASH (confirmed from pipeline.log):
═══════════════════════════════════════════════════════════════
  Crash sequence:
    01:57:42  SRT written successfully (Whisper finished)
    01:57:42  Step 2/4 — Building video segments
    01:57:42  Building title card…
    [PROCESS DIES — no Python exception, no traceback]

  Cause:
    whisper.load_model() imports PyTorch, which initialises the CUDA
    runtime INSIDE the main uvicorn process.  After `del model`, Python's
    garbage collector tries to release the torch CUDA tensors on a
    background GC thread.  On Windows, the CUDA driver performs teardown
    while the CUDA context is already partially freed → C-level access
    violation → the OS kills the entire process instantly.
    Python try/except cannot catch C-level crashes; that is why the server
    dies with no traceback.

  Fix (this file):
    Whisper is called via subprocess.run([sys.executable, "_whisper_worker.py",
    audio, srt, model]).  The child process owns the CUDA context.  When the
    child exits normally, the OS reclaims ALL GPU/CUDA memory cleanly before
    the parent continues.  The parent process never sees a torch object.

SUBTITLE PATH FIX (WinError / spaces):
═══════════════════════════════════════
  The project lives in "D:\\Ai Agents\\YT Automation" — the space in
  "Ai Agents" caused FFmpeg's subtitles= filter to silently reject the
  path and produce a video with no subtitles.  The fix copies the SRT
  file to %TEMP%\\yt_agent_{run_id}.srt (no spaces) before passing it
  to FFmpeg, then removes the copy afterwards.

OTHER RULES ENFORCED HERE:
  * No MoviePy import anywhere — OOM-killed on 6 GB VRAM systems.
  * No numpy import at module level (torch pulls it in on some builds).
  * PIL used only for title/outro PNG creation (pure-CPU, <10 MB).
  * Every FFmpeg call is a subprocess — never holds video in RAM.
  * Segments written to disk one at a time.
  * Concat uses FFmpeg concat demuxer (~0 extra RAM, streaming).
  * zoompan Ken-Burns done inside FFmpeg (GPU-optional).
"""
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

import requests

from agents.base_agent import BaseAgent
import config

logger = logging.getLogger(__name__)

W   = config.VIDEO_WIDTH    # 1920
H   = config.VIDEO_HEIGHT   # 1080
FPS = 24

# Absolute path to the worker script (project root)
_WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_whisper_worker.py",
)


class VideoAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="VideoAgent",
            role_description=(
                "Assembles the final video from voiceover audio, visual assets, and subtitles. "
                "Whisper is isolated in a child process to prevent CUDA GC crashes. "
                "All video operations use FFmpeg subprocesses only."
            ),
            inputs_expected=["voiceover_data", "visual_data", "script_data", "run_id"],
            outputs_produced=["video_data"],
        )

    # ─────────────────────────────────────────────────────────────────────
    #  FFmpeg helpers
    # ─────────────────────────────────────────────────────────────────────

    def _run_ffmpeg(self, cmd: list, run_id: str, label: str = "",
                    timeout: int = 600) -> bool:
        """Run one FFmpeg command. Returns True on success."""
        try:
            self._log(run_id,
                      f"FFmpeg [{label}]: {' '.join(str(c) for c in cmd[:7])}…",
                      "INFO")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                self._log(run_id,
                          f"FFmpeg [{label}] stderr:\n{result.stderr[-600:]}",
                          "WARNING")
                return False
            return True
        except subprocess.TimeoutExpired:
            self._log(run_id, f"FFmpeg [{label}] timed out after {timeout}s", "WARNING")
            return False
        except FileNotFoundError:
            self._log(run_id, "FFmpeg not found in PATH. Install FFmpeg.", "ERROR")
            return False
        except Exception as e:
            self._log(run_id, f"FFmpeg [{label}] exception: {e}", "WARNING")
            return False

    def _get_video_duration(self, path: str) -> float:
        """Return duration in seconds via ffprobe, or 0 on failure."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", path],
                capture_output=True, text=True, timeout=30,
            )
            data = json.loads(result.stdout)
            for stream in data.get("streams", []):
                dur = stream.get("duration")
                if dur:
                    return float(dur)
        except Exception:
            pass
        return 0.0

    def _download_file(self, url: str, dest: str, run_id: str) -> bool:
        """Download a URL to a local path; returns True on success."""
        try:
            resp = requests.get(url, timeout=(5, 60), stream=True)
            if resp.status_code != 200:
                self._log(run_id, f"HTTP {resp.status_code} for {url[:60]}", "WARNING")
                return False
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(8192):
                    fh.write(chunk)
            size = os.path.getsize(dest) if os.path.exists(dest) else 0
            return size > 512
        except Exception as e:
            self._log(run_id, f"Download failed ({url[:60]}): {e}", "WARNING")
            return False

    # ─────────────────────────────────────────────────────────────────────
    #  CRASH FIX: Whisper runs in an isolated child process
    # ─────────────────────────────────────────────────────────────────────

    def _transcribe_to_srt(self, audio_path: str, srt_path: str,
                           run_id: str) -> bool:
        """
        Transcribe audio → SRT by spawning _whisper_worker.py as a subprocess.

        WHY: whisper.load_model() pulls in PyTorch and initialises the CUDA
        runtime inside the calling process.  On Windows, when Python's GC later
        tries to destroy those CUDA tensors on a background thread, the CUDA
        driver performs teardown while the context is already partially freed
        → C-level access violation → the OS kills the entire process.
        Running Whisper in a child process means the child owns the CUDA
        context; when the child exits the OS reclaims everything atomically.
        """
        if not os.path.exists(_WORKER_SCRIPT):
            self._log(run_id, f"Whisper worker not found: {_WORKER_SCRIPT}", "WARNING")
            return False

        self._log(run_id,
                  f"Launching Whisper child process (model={config.WHISPER_MODEL})…",
                  "INFO")
        try:
            result = subprocess.run(
                [sys.executable, _WORKER_SCRIPT,
                 audio_path, srt_path, config.WHISPER_MODEL],
                capture_output=True, text=True, timeout=300,
            )
            for line in result.stdout.strip().splitlines():
                if line:
                    self._log(run_id, f"[whisper] {line}", "INFO")
            if result.returncode != 0:
                self._log(run_id,
                          f"Whisper child exited {result.returncode}: "
                          f"{result.stderr[-400:]}",
                          "WARNING")
                return False
            if not os.path.exists(srt_path) or os.path.getsize(srt_path) < 5:
                self._log(run_id, "SRT file missing/empty after Whisper", "WARNING")
                return False
            self._log(run_id, f"SRT ready: {srt_path}", "INFO")
            return True
        except subprocess.TimeoutExpired:
            self._log(run_id, "Whisper child timed out (300 s)", "WARNING")
            return False
        except Exception as e:
            self._log(run_id, f"Whisper subprocess error: {e}", "WARNING")
            return False

    # ─────────────────────────────────────────────────────────────────────
    #  PIL card creators (pure-CPU, no torch, no numpy)
    # ─────────────────────────────────────────────────────────────────────

    def _load_font(self, size: int):
        """Return the best available PIL font at *size* pt, or default."""
        from PIL import ImageFont
        for fp in [config.THUMBNAIL_FONT_PATH, config.FALLBACK_FONT_PATH]:
            try:
                if fp and os.path.exists(fp) and os.path.getsize(fp) > 0:
                    return ImageFont.truetype(fp, size)
            except Exception:
                pass
        for sys_fp in [
            "C:/Windows/Fonts/impact.ttf",
            "C:/Windows/Fonts/ariblk.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]:
            try:
                if os.path.exists(sys_fp):
                    return ImageFont.truetype(sys_fp, size)
            except Exception:
                pass
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    def _wrap_text(self, draw, text: str, font, max_width: int) -> list:
        words, lines, current = text.split(), [], ""
        for word in words:
            test = (current + " " + word).strip()
            try:
                tw = draw.textbbox((0, 0), test, font=font)[2]
            except Exception:
                tw = len(test) * max(10, getattr(font, "size", 40))
            if tw > max_width and current:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)
        return lines or [text]

    def _create_title_card_png(self, title: str, out_path: str) -> bool:
        """Safe title card — uses default font to prevent C-level freetype issues."""
        try:
            from PIL import Image as PILImage, ImageDraw, ImageFont
            img  = PILImage.new("RGB", (W, H), (20, 25, 45))
            draw = ImageDraw.Draw(img)
            for y in range(H):
                r = int(20 + 20 * (y / H))
                g = int(25 + 15 * (y / H))
                b = int(45 + 40 * (y / H))
                draw.line([(0, y), (W, y)], fill=(r, g, b))
            font_safe = ImageFont.load_default()
            draw.text((150, H // 2 - 30), f"TOPIC: {title}", fill="white",  font=font_safe)
            draw.text((150, H // 2 + 30), "Powered by AI",  fill="gray",   font=font_safe)
            img.save(out_path, "PNG")
            return True
        except Exception as e:
            self._log("unknown", f"Title card failed: {e}", "ERROR")
            return False

    def _create_outro_card_png(self, out_path: str) -> bool:
        """Safe outro card."""
        try:
            from PIL import Image as PILImage, ImageDraw, ImageFont
            img  = PILImage.new("RGB", (W, H), (140, 15, 15))
            draw = ImageDraw.Draw(img)
            for y in range(H):
                r = int(160 - 50 * (abs(y - H // 2) / (H // 2)))
                draw.line([(0, y), (W, y)], fill=(r, 15, 15))
            font_safe = ImageFont.load_default()
            draw.text((W // 2 - 60, H // 2 - 10), "SUBSCRIBE", fill="white", font=font_safe)
            img.save(out_path, "PNG")
            return True
        except Exception as e:
            self._log("unknown", f"Outro card failed: {e}", "ERROR")
            return False

    def _create_gradient_png(self, out_path: str, idx: int = 0) -> bool:
        """Coloured gradient placeholder PNG."""
        palette = [
            ((20, 30, 60),  (40, 60, 120)),
            ((30, 20, 60),  (60, 40, 120)),
            ((20, 40, 40),  (40, 80, 80)),
            ((40, 25, 20),  (80, 50, 40)),
        ]
        c1, c2 = palette[idx % len(palette)]
        try:
            from PIL import Image, ImageDraw
            img  = Image.new("RGB", (W, H))
            draw = ImageDraw.Draw(img)
            for y in range(H):
                t = y / H
                draw.line(
                    [(0, y), (W, y)],
                    fill=(
                        int(c1[0] + (c2[0] - c1[0]) * t),
                        int(c1[1] + (c2[1] - c1[1]) * t),
                        int(c1[2] + (c2[2] - c1[2]) * t),
                    ),
                )
            img.save(out_path, "PNG")
            return True
        except Exception as e:
            logger.warning("Gradient PNG failed: %s", e)
            return False

    # ─────────────────────────────────────────────────────────────────────
    #  FFmpeg segment builders
    # ─────────────────────────────────────────────────────────────────────

    def _image_to_video_segment(self, image_path: str, duration: float,
                                out_path: str, run_id: str,
                                use_ken_burns: bool = True) -> bool:
        """Convert a PNG/JPG to a fixed-duration H.264 segment via FFmpeg."""
        duration = max(duration, 1.0)
        frames   = int(duration * FPS)

        if use_ken_burns and config.KEN_BURNS_ZOOM > 0:
            zoom_end   = 1.0 + config.KEN_BURNS_ZOOM
            zoom_delta = config.KEN_BURNS_ZOOM / max(frames, 1)
            zoom_expr  = f"min(zoom+{zoom_delta:.7f},{zoom_end:.5f})"
            scale_up   = f"{int(W * 1.15)}:{int(H * 1.15)}"
            vf = (
                f"scale={scale_up}:flags=lanczos,"
                f"zoompan=z='{zoom_expr}'"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":d={frames}:s={W}x{H}:fps={FPS},"
                f"scale={W}:{H}:flags=lanczos"
            )
        else:
            vf = (
                f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black"
            )
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", image_path,
            "-vf", vf, "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-r", str(FPS), "-an",
            out_path,
        ]
        ok = self._run_ffmpeg(cmd, run_id, f"img2vid({os.path.basename(image_path)[:30]})")
        if ok and os.path.exists(out_path) and os.path.getsize(out_path) < 1000:
            self._log(run_id, "img2vid produced tiny file — treating as failure", "WARNING")
            return False
        return ok

    def _video_to_segment(self, video_path: str, duration: float,
                          out_path: str, run_id: str) -> bool:
        """Normalise a video file to 1920×1080 24fps H.264 exact-duration."""
        duration = max(duration, 1.0)
        vf = (
            f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={W}:{H}"
        )
        cmd = [
            "ffmpeg", "-y", "-stream_loop", "-1", "-i", video_path,
            "-vf", vf, "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-r", str(FPS), "-an",
            out_path,
        ]
        ok = self._run_ffmpeg(cmd, run_id, f"vid2seg({os.path.basename(video_path)[:30]})")
        if ok and os.path.exists(out_path) and os.path.getsize(out_path) < 1000:
            self._log(run_id, "vid2seg produced tiny file — treating as failure", "WARNING")
            return False
        return ok

    # ─────────────────────────────────────────────────────────────────────
    #  Per-section segment builder
    # ─────────────────────────────────────────────────────────────────────

    def _build_segment(self, item: dict, seg_idx: int,
                       out_path: str, tmp_dir: str, run_id: str) -> bool:
        """Build one video segment from a visual_map entry."""
        start      = float(item.get("start_time", 0))
        end        = float(item.get("end_time", 30))
        duration   = max(end - start, 1.5)
        asset_path = item.get("asset_path", "")
        asset_type = item.get("asset_type", "image")
        title      = item.get("section_title", f"Section {seg_idx}")

        self._log(run_id,
                  f"  Seg {seg_idx:02d}: '{title[:40]}' "
                  f"{duration:.1f}s [{asset_type}] "
                  f"{os.path.basename(str(asset_path))[:40]}",
                  "INFO")

        local_path = asset_path

        # Download remote URLs
        if isinstance(asset_path, str) and asset_path.startswith("http"):
            ext    = ".mp4" if asset_type == "video" else ".jpg"
            dl_dst = os.path.join(tmp_dir, f"dl_{seg_idx:03d}{ext}")
            if self._download_file(asset_path, dl_dst, run_id):
                local_path = dl_dst
            else:
                self._log(run_id, f"  Download failed for seg {seg_idx}", "WARNING")
                local_path = ""

        # Route to correct encoder
        if (asset_type == "video" and local_path
                and os.path.exists(str(local_path))
                and os.path.getsize(str(local_path)) > 1000):
            if self._video_to_segment(str(local_path), duration, out_path, run_id):
                return True
            self._log(run_id, f"  Video encode failed for seg {seg_idx}", "WARNING")

        if (local_path and os.path.exists(str(local_path))
                and os.path.getsize(str(local_path)) > 512):
            if self._image_to_video_segment(str(local_path), duration, out_path, run_id):
                return True

        # Gradient placeholder
        self._log(run_id, f"  Using gradient placeholder for seg {seg_idx}", "WARNING")
        grad = os.path.join(tmp_dir, f"grad_{seg_idx:03d}.png")
        if self._create_gradient_png(grad, seg_idx):
            return self._image_to_video_segment(grad, duration, out_path,
                                                run_id, use_ken_burns=False)
        return False

    # ─────────────────────────────────────────────────────────────────────
    #  Concat
    # ─────────────────────────────────────────────────────────────────────

    def _write_concat_list(self, segment_paths: list, list_path: str):
        with open(list_path, "w", encoding="utf-8") as fh:
            for p in segment_paths:
                fh.write(f"file '{p.replace(chr(92), '/')}'\n")

    def _concat_segments(self, segment_paths: list, out_path: str,
                         run_id: str) -> bool:
        """Join segments using FFmpeg concat demuxer (~0 extra RAM)."""
        if not segment_paths:
            return False

        list_path = out_path.replace(".mp4", "_list.txt")
        self._write_concat_list(segment_paths, list_path)

        # Try crossfade if configured
        if config.CROSSFADE_DURATION > 0 and len(segment_paths) > 1:
            if self._concat_with_xfade(segment_paths, out_path, run_id):
                try:
                    os.remove(list_path)
                except Exception:
                    pass
                return True

        # Simple concat
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-r", str(FPS), "-an",
            out_path,
        ]
        ok = self._run_ffmpeg(cmd, run_id, "concat-simple", timeout=900)
        try:
            os.remove(list_path)
        except Exception:
            pass
        return ok

    def _concat_with_xfade(self, segment_paths: list, out_path: str,
                            run_id: str) -> bool:
        """Crossfade using FFmpeg xfade filter."""
        fade = config.CROSSFADE_DURATION
        n    = len(segment_paths)
        try:
            durations = [self._get_video_duration(p) or 5.0 for p in segment_paths]
            inputs    = []
            for p in segment_paths:
                inputs += ["-i", p]
            filters, prev, cum = [], "[0:v]", 0.0
            for i in range(1, n):
                cum += durations[i - 1] - fade
                lbl  = f"[v{i}]" if i < n - 1 else "[vout]"
                filters.append(
                    f"{prev}[{i}:v]xfade=transition=fade"
                    f":duration={fade}:offset={cum:.3f}{lbl}"
                )
                prev = f"[v{i}]"
            cmd = [
                "ffmpeg", "-y", *inputs,
                "-filter_complex", ";".join(filters),
                "-map", "[vout]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-pix_fmt", "yuv420p", "-r", str(FPS), "-an",
                out_path,
            ]
            ok = self._run_ffmpeg(cmd, run_id, "concat-xfade", timeout=900)
            if ok and os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
                return True
        except Exception as e:
            self._log(run_id, f"xfade error ({e}); falling back to simple concat", "WARNING")
        return False

    # ─────────────────────────────────────────────────────────────────────
    #  Subtitle path helper — FIX: copy to %TEMP% to avoid spaces
    # ─────────────────────────────────────────────────────────────────────

    def _safe_srt_path(self, srt_path: str, run_id: str) -> str:
        """
        Return a path to the SRT file that has NO spaces and uses forward
        slashes, safe for FFmpeg's subtitles= filter on Windows.

        The project root is 'D:\\Ai Agents\\YT Automation' — the space
        in 'Ai Agents' causes FFmpeg to parse the subtitles filter argument
        incorrectly on Windows, producing a video with no subtitles (or
        failing the whole mux).

        Fix: copy the SRT to %TEMP% under a short name, return that path.
        """
        if " " not in srt_path:
            # No spaces — just escape the colon for FFmpeg's filter syntax
            esc = srt_path.replace("\\", "/")
            if len(esc) > 1 and esc[1] == ":":
                esc = esc[0] + "\\:" + esc[2:]
            return esc

        # Copy to TEMP (always has no spaces in the filename we generate)
        tmp_srt = os.path.join(
            tempfile.gettempdir(),
            f"yt_agent_{run_id[:8]}.srt",
        )
        try:
            shutil.copy2(srt_path, tmp_srt)
            self._log(run_id,
                      f"SRT copied to temp (spaces in original path): {tmp_srt}",
                      "INFO")
            esc = tmp_srt.replace("\\", "/")
            if len(esc) > 1 and esc[1] == ":":
                esc = esc[0] + "\\:" + esc[2:]
            return esc
        except Exception as e:
            self._log(run_id, f"SRT temp-copy failed ({e}); subtitles may be skipped", "WARNING")
            # Last resort: just escape what we have and hope for the best
            esc = srt_path.replace("\\", "/")
            if len(esc) > 1 and esc[1] == ":":
                esc = esc[0] + "\\:" + esc[2:]
            return esc

    # ─────────────────────────────────────────────────────────────────────
    #  Audio + subtitle mux
    # ─────────────────────────────────────────────────────────────────────

    def _add_audio_and_subtitles(self, video_path: str, audio_path: str,
                                 srt_path: str, out_path: str,
                                 run_id: str) -> str:
        """Mux voiceover + optional BG music + optional subtitles."""
        has_srt = os.path.exists(srt_path) and os.path.getsize(srt_path) > 10
        has_bg  = os.path.exists(config.BG_MUSIC_PATH)

        if has_srt:
            safe_srt = self._safe_srt_path(srt_path, run_id)
            sub_vf = (
                f"subtitles='{safe_srt}'"
                ":force_style='FontName=Arial,FontSize=18,"
                "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                "Outline=2,Shadow=1,Bold=1,Alignment=2,MarginV=40'"
            )
            self._log(run_id, f"Subtitle filter path: {safe_srt}", "INFO")
        else:
            sub_vf = None

        if has_bg:
            afilt = (
                f"[1:a]volume=1.0[vo];"
                f"[2:a]volume={config.BG_MUSIC_VOLUME},"
                f"aloop=loop=-1:size=2e+09[bg];"
                f"[vo][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
            extra = ["-i", audio_path, "-i", config.BG_MUSIC_PATH]
            amap  = ["-map", "[aout]"]
        else:
            afilt = None
            extra = ["-i", audio_path]
            amap  = ["-map", "1:a"]

        parts = []
        if afilt:
            parts.append(afilt)
        if sub_vf:
            parts.append(f"[0:v]{sub_vf}[vout]")
            vmap = ["-map", "[vout]"]
        else:
            vmap = ["-map", "0:v"]

        cmd = ["ffmpeg", "-y", "-i", video_path] + extra
        if parts:
            cmd += ["-filter_complex", ";".join(parts)]
        cmd += vmap + amap + [
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "192k", "-shortest", out_path,
        ]
        ok = self._run_ffmpeg(cmd, run_id, "mux-audio-subs", timeout=900)
        if ok and os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
            # Clean up temp SRT if we created one
            tmp_srt = os.path.join(
                tempfile.gettempdir(), f"yt_agent_{run_id[:8]}.srt"
            )
            if os.path.exists(tmp_srt):
                try:
                    os.remove(tmp_srt)
                except Exception:
                    pass
            return out_path

        self._log(run_id, "Full mux failed — retrying without subtitles", "WARNING")
        cmd2 = [
            "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "192k", "-shortest", out_path,
        ]
        ok2 = self._run_ffmpeg(cmd2, run_id, "mux-audio-only", timeout=600)
        return out_path if (ok2 and os.path.exists(out_path)) else video_path

    # ─────────────────────────────────────────────────────────────────────
    #  Main execute
    # ─────────────────────────────────────────────────────────────────────

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        run_id         = state["run_id"]
        voiceover_data = state["voiceover_data"]
        visual_data    = state["visual_data"]
        trend_data     = state.get("trend_data", {})

        audio_path = voiceover_data.get("audio_path", "")
        visual_map = visual_data.get("visual_map", [])
        topic      = trend_data.get("topic", "AI Video")

        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not visual_map:
            raise ValueError("visual_map is empty — VisualAgent must run first.")

        run_output_dir = os.path.join(config.OUTPUT_DIR, f"run_{run_id}")
        os.makedirs(run_output_dir, exist_ok=True)
        tmp_dir = os.path.join(run_output_dir, "tmp_segs")
        os.makedirs(tmp_dir, exist_ok=True)

        video_clips = sum(1 for v in visual_map if v.get("asset_type") == "video")
        image_clips = len(visual_map) - video_clips
        self._log(run_id,
                  f"Visual map: {video_clips} video + {image_clips} image clips. "
                  f"Crossfade={config.CROSSFADE_DURATION}s  KenBurns={config.KEN_BURNS_ZOOM}",
                  "INFO")

        # ── Step 1: Whisper in isolated child process (CRASH FIX) ─────
        srt_path = os.path.join(run_output_dir, "subtitles.srt")
        self._log(run_id, "Step 1/4: Whisper transcription (isolated child process)…", "INFO")
        srt_ok = self._transcribe_to_srt(audio_path, srt_path, run_id)
        if not srt_ok:
            self._log(run_id, "SRT failed — video will have no subtitles", "WARNING")

        # GC before FFmpeg assembly to clear any lingering LLM objects
        gc.collect()
        self._log(run_id, "GC done. Starting FFmpeg-only assembly…", "INFO")

        # ── Step 2: Build individual segments ─────────────────────────
        self._log(run_id, "Step 2/4: Building video segments…", "INFO")
        segment_paths: List[str] = []

        # Title card
        self._log(run_id, "  Building title card…", "INFO")
        title_png = os.path.join(tmp_dir, "title_card.png")
        title_seg = os.path.join(tmp_dir, "seg_title.mp4")
        if self._create_title_card_png(topic, title_png):
            if self._image_to_video_segment(title_png, 2.0, title_seg,
                                            run_id, use_ken_burns=False):
                segment_paths.append(title_seg)
                self._log(run_id, "  Title card: OK", "INFO")
            else:
                self._log(run_id, "  Title card FFmpeg step failed (skipped)", "WARNING")
        else:
            self._log(run_id, "  Title card PNG failed (skipped)", "WARNING")

        # Body segments
        for i, item in enumerate(visual_map):
            seg_out = os.path.join(tmp_dir, f"seg_{i:03d}.mp4")
            ok      = self._build_segment(item, i, seg_out, tmp_dir, run_id)
            if ok and os.path.exists(seg_out) and os.path.getsize(seg_out) > 1000:
                segment_paths.append(seg_out)
            else:
                self._log(run_id, f"  Seg {i} failed — gradient fallback", "WARNING")
                dur    = max(
                    float(item.get("end_time", 30)) - float(item.get("start_time", 0)),
                    1.5,
                )
                gb_png = os.path.join(tmp_dir, f"grad_fb_{i:03d}.png")
                fb_seg = os.path.join(tmp_dir, f"seg_fb_{i:03d}.mp4")
                if (self._create_gradient_png(gb_png, i)
                        and self._image_to_video_segment(gb_png, dur, fb_seg, run_id, False)):
                    segment_paths.append(fb_seg)

        # Outro card
        self._log(run_id, "  Building outro card…", "INFO")
        outro_png = os.path.join(tmp_dir, "outro_card.png")
        outro_seg = os.path.join(tmp_dir, "seg_outro.mp4")
        if self._create_outro_card_png(outro_png):
            if self._image_to_video_segment(outro_png, 3.0, outro_seg,
                                            run_id, use_ken_burns=False):
                segment_paths.append(outro_seg)
                self._log(run_id, "  Outro card: OK", "INFO")

        if not segment_paths:
            raise RuntimeError(
                "All segments failed — check FFmpeg installation and asset URLs."
            )
        self._log(run_id, f"  {len(segment_paths)} segments ready.", "INFO")

        # ── Step 3: Concatenate ────────────────────────────────────────
        self._log(run_id, "Step 3/4: Concatenating segments…", "INFO")
        concat_path = os.path.join(run_output_dir, "video_concat.mp4")
        if not self._concat_segments(segment_paths, concat_path, run_id):
            raise RuntimeError(
                "FFmpeg concat failed. Ensure FFmpeg >= 4.4 is in PATH."
            )

        # ── Step 4: Mux audio + subtitles ─────────────────────────────
        self._log(run_id, "Step 4/4: Muxing audio and subtitles…", "INFO")
        final_path = os.path.join(run_output_dir, "final_video.mp4")
        final_path = self._add_audio_and_subtitles(
            concat_path, audio_path, srt_path, final_path, run_id
        )

        # Cleanup temp segments
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if concat_path != final_path and os.path.exists(concat_path):
                os.remove(concat_path)
        except Exception:
            pass

        duration = (
            self._get_video_duration(final_path)
            or voiceover_data.get("duration_seconds", 0)
        )
        self._log(run_id,
                  f"VideoAgent complete ✓  {final_path} ({duration:.1f}s, {W}x{H})",
                  "INFO")

        return {
            "video_data": {
                "video_path":     final_path,
                "duration":       round(duration, 2),
                "resolution":     f"{W}x{H}",
                "has_subtitles":  srt_ok and os.path.exists(srt_path),
                "srt_path":       srt_path,
                "clip_breakdown": {
                    "video_clips": video_clips,
                    "image_clips": image_clips,
                },
            }
        }
