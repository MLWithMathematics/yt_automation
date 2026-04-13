"""
agents/visual_agent.py — Agent 5: B-Roll VIDEO fetching (v3 — AnimateDiff Edition)

Priority chain per script section:
  1. Pexels Videos API   → real HD stock video clip (trimmed to exact duration)
  2. AnimateDiff (GPU)   → text-to-video, cinematic AI clip (512px → upscaled to 1080p)
  3. SVD (GPU, optional) → animates a Pexels photo into motion video
  4. Pexels Photos API   → static image (Ken Burns applied in VideoAgent)
  5. Pollinations.ai     → AI-generated image (Ken Burns fallback, no key needed)

RTX 3050 6GB notes:
  - AnimateDiff: ~5GB VRAM with fp16 + CPU offload, ~1-3 min/clip, RECOMMENDED
  - SVD: ~5.5GB VRAM with CPU offload, ~2-5 min/clip (disable in config if slow)
  - Both use ANIMATEDIFF_UNLOAD_AFTER_CLIP=True to free VRAM between clips
  - Short videos (2min): AnimateDiff feasible for all sections
  - Medium videos (7min): Mix Pexels video + AnimateDiff for unmatched sections
  - Long videos (12min): Prefer Pexels video, use AnimateDiff sparingly
"""
import json
import logging
import os
import subprocess
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import requests
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
import config

logger = logging.getLogger(__name__)

PEXELS_HEADERS = {"Authorization": config.PEXELS_API_KEY}


class VisualAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="VisualAgent",
            role_description=(
                "Fetches realistic video clips for each script section. "
                "Priority: Pexels HD Videos → AnimateDiff text-to-video → "
                "SVD animated image → Pexels Photos → Pollinations.ai images."
            ),
            inputs_expected=["script_data", "voiceover_data", "trend_data"],
            outputs_produced=["visual_data"],
        )
        self._animatediff_pipe = None   # lazy-loaded on first use
        self._svd_pipe = None           # lazy-loaded on first use

    # ────────────────────────────────────────────────────────────────────
    #  LLM: Visual search query generation
    # ────────────────────────────────────────────────────────────────────
    def _generate_visual_query(self, section: dict, topic: str, run_id: str) -> str:
        """Ask the LLM for the best visual search query for this section."""
        system_prompt = (
            "You are a cinematographer. Given a script section, output ONE short "
            "visual description (3-7 words) perfect for finding a stock VIDEO clip. "
            "Think: action, movement, real locations, people doing things. "
            "Return ONLY the description — no quotes, no punctuation."
        )
        user_prompt = (
            f"Topic: {topic}\n"
            f"Section: {section.get('title', '')}\n"
            f"Content preview: {section.get('content', '')[:200]}\n\n"
            "Cinematographic search query:"
        )
        try:
            result = self.call_llm_with_retry(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
                run_id=run_id,
            )
            return result.strip().strip('"').strip("'")[:100]
        except Exception as e:
            self._log(run_id, f"Visual query generation failed: {e}", "WARNING")
            return topic

    def _generate_animatediff_prompt(self, query: str, section: dict, run_id: str) -> str:
        """Generate a rich cinematic prompt for AnimateDiff text-to-video."""
        system_prompt = (
            "You are a film director writing text-to-video prompts for an AI model. "
            "Write a single vivid sentence describing a SHORT cinematic video clip. "
            "Include: subject, action, lighting style, camera movement, mood. "
            "Format: '[subject] [action], [lighting], [camera], [mood]'. "
            "Example: 'A scientist examining glowing data on a holographic screen, "
            "blue neon lighting, slow camera push-in, futuristic atmosphere'. "
            "Return ONLY the prompt, no extra text."
        )
        user_prompt = (
            f"Base query: {query}\n"
            f"Section title: {section.get('title', '')}\n"
            "Cinematic video prompt:"
        )
        try:
            result = self.call_llm_with_retry(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
                run_id=run_id,
            )
            return result.strip()[:300]
        except Exception as e:
            self._log(run_id, f"AnimateDiff prompt generation failed: {e}", "WARNING")
            return f"cinematic footage of {query}, professional cinematography, smooth motion"

    # ────────────────────────────────────────────────────────────────────
    #  LAYER 1: Pexels Videos (real stock HD video)
    # ────────────────────────────────────────────────────────────────────
    def _fetch_pexels_video_url(self, query: str, needed_duration: float, run_id: str) -> Optional[str]:
        """Search Pexels Videos API. Returns best download URL or None."""
        if not config.PEXELS_API_KEY:
            return None
        try:
            resp = requests.get(
                "https://api.pexels.com/videos/search",
                headers={"Authorization": config.PEXELS_API_KEY},
                params={
                    "query": query,
                    "orientation": "landscape",
                    "size": "medium",       # medium = 1280x720+, large = 1920x1080+
                    "per_page": config.PEXELS_PER_SECTION,
                },
                timeout=20,
            )
            if resp.status_code == 429:
                self._log(run_id, "Pexels Videos quota hit", "WARNING")
                return None
            if resp.status_code != 200:
                self._log(run_id, f"Pexels Videos HTTP {resp.status_code}", "WARNING")
                return None

            videos = resp.json().get("videos", [])
            if not videos:
                return None

            # Score: prefer clips close to or longer than needed_duration
            def score(v):
                dur = v.get("duration", 0)
                return abs(dur - needed_duration) if dur >= needed_duration else 1000 + (needed_duration - dur)

            for video in sorted(videos, key=score):
                files = video.get("video_files", [])
                hd_files = sorted(
                    [f for f in files if f.get("quality") in ("hd", "sd")],
                    key=lambda f: f.get("width", 0),
                    reverse=True,
                )
                if hd_files:
                    link = hd_files[0].get("link")
                    if link:
                        self._log(
                            run_id,
                            f"  Pexels video: {video['duration']}s "
                            f"{hd_files[0].get('width')}x{hd_files[0].get('height')}",
                            "INFO",
                        )
                        return link
        except Exception as e:
            self._log(run_id, f"Pexels Videos error: {e}", "WARNING")
        return None

    def _download_video(self, url: str, save_path: str, run_id: str) -> bool:
        """Stream-download a video file."""
        try:
            resp = requests.get(url, timeout=120, stream=True)
            if resp.status_code != 200:
                self._log(run_id, f"Video download HTTP {resp.status_code}", "WARNING")
                return False
            total = 0
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(65536):
                    f.write(chunk)
                    total += len(chunk)
            if total < 10_000:
                self._log(run_id, f"Downloaded file too small ({total} bytes)", "WARNING")
                return False
            self._log(run_id, f"  Downloaded {total/1_048_576:.1f} MB → {save_path}", "INFO")
            return True
        except Exception as e:
            self._log(run_id, f"Video download error: {e}", "WARNING")
            return False

    def _trim_video_clip(self, src: str, out: str, duration: float, run_id: str) -> bool:
        """
        FFmpeg: trim/loop source to exactly `duration` seconds,
        scale to VIDEO_WIDTH x VIDEO_HEIGHT, 24fps, no audio.
        """
        W, H = config.VIDEO_WIDTH, config.VIDEO_HEIGHT
        try:
            # Probe source duration
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", src],
                capture_output=True, text=True, timeout=30,
            )
            src_dur = 0.0
            try:
                for stream in json.loads(probe.stdout).get("streams", []):
                    d = float(stream.get("duration", 0))
                    if d > src_dur:
                        src_dur = d
            except Exception:
                src_dur = duration

            if src_dur < 0.5:
                self._log(run_id, f"Source clip too short ({src_dur:.1f}s)", "WARNING")
                return False

            scale_filter = (
                f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
            )

            if src_dur < duration:
                loop_count = int(duration / src_dur) + 2
                cmd = [
                    "ffmpeg", "-y",
                    "-stream_loop", str(loop_count),
                    "-i", src,
                    "-t", str(duration),
                    "-vf", scale_filter,
                    "-r", "24", "-an",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    out,
                ]
            else:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", src,
                    "-t", str(duration),
                    "-vf", scale_filter,
                    "-r", "24", "-an",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    out,
                ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                self._log(run_id, f"FFmpeg trim error: {result.stderr[:400]}", "WARNING")
                return False
            return True
        except subprocess.TimeoutExpired:
            self._log(run_id, "FFmpeg trim timed out", "WARNING")
            return False
        except Exception as e:
            self._log(run_id, f"Video trim error: {e}", "WARNING")
            return False

    # ────────────────────────────────────────────────────────────────────
    #  LAYER 2: AnimateDiff — text-to-video (FREE, open-source)
    # ────────────────────────────────────────────────────────────────────
    def _check_animatediff_available(self, run_id: str) -> bool:
        """Check GPU + library availability for AnimateDiff."""
        if not config.ENABLE_ANIMATEDIFF:
            return False
        try:
            import torch
            if not torch.cuda.is_available():
                self._log(run_id, "AnimateDiff: No CUDA GPU detected", "WARNING")
                return False
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram < 4.5:
                self._log(run_id, f"AnimateDiff: VRAM too low ({vram:.1f}GB, need 4.5GB+)", "WARNING")
                return False
            import diffusers  # noqa
            self._log(
                run_id,
                f"AnimateDiff available: {torch.cuda.get_device_name(0)} ({vram:.1f}GB)",
                "INFO",
            )
            return True
        except ImportError:
            self._log(
                run_id,
                "AnimateDiff: diffusers not installed. "
                "Run: pip install diffusers transformers accelerate",
                "WARNING",
            )
            return False
        except Exception as e:
            self._log(run_id, f"AnimateDiff availability check failed: {e}", "WARNING")
            return False

    def _generate_animatediff_video(
        self, prompt: str, out_path: str, duration: float, run_id: str
    ) -> bool:
        """
        Generate a short cinematic video clip using AnimateDiff + SD 1.5.

        Model: guoyww/animatediff-motion-adapter-v1-5-2
        Base:  runwayml/stable-diffusion-v1-5
        VRAM:  ~5GB with fp16 + model_cpu_offload + vae_slicing
        Time:  ~1-3 minutes on RTX 3050 per clip
        Output: 16 frames @ 8fps = 2s, upscaled to 1920x1080 via FFmpeg
        """
        try:
            import torch
            from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
            from diffusers.utils import export_to_video

            self._log(run_id, f"AnimateDiff generating: '{prompt[:80]}...'", "INFO")

            # Load pipeline (cache between clips unless UNLOAD_AFTER_CLIP is set)
            if self._animatediff_pipe is None:
                self._log(run_id, "Loading AnimateDiff pipeline (first run downloads ~4GB)...", "INFO")
                adapter = MotionAdapter.from_pretrained(
                    config.ANIMATEDIFF_ADAPTER,
                    torch_dtype=torch.float16,
                )
                pipe = AnimateDiffPipeline.from_pretrained(
                    config.ANIMATEDIFF_BASE_MODEL,
                    motion_adapter=adapter,
                    torch_dtype=torch.float16,
                )
                # DDIMScheduler config recommended for AnimateDiff v1.5
                pipe.scheduler = DDIMScheduler.from_config(
                    pipe.scheduler.config,
                    beta_schedule="linear",
                    clip_sample=False,
                    timestep_spacing="linspace",
                    steps_offset=1,
                )
                # Memory optimizations for 6GB VRAM
                pipe.enable_model_cpu_offload()   # offload layers to CPU when idle
                pipe.enable_vae_slicing()          # decode VAE in slices → less peak VRAM
                self._animatediff_pipe = pipe
                self._log(run_id, "AnimateDiff pipeline loaded (fp16 + CPU offload)", "INFO")
            else:
                pipe = self._animatediff_pipe

            negative_prompt = (
                "blurry, static, still image, low quality, watermark, text, "
                "oversaturated, bad anatomy, distorted, cartoon"
            )

            # Enhance prompt for cinematic quality
            full_prompt = (
                f"cinematic 4K professional footage, {prompt}, "
                "smooth camera movement, high detail, photorealistic"
            )

            with torch.autocast("cuda", dtype=torch.float16):
                output = pipe(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,
                    num_frames=config.ANIMATEDIFF_NUM_FRAMES,
                    guidance_scale=config.ANIMATEDIFF_GUIDANCE,
                    num_inference_steps=config.ANIMATEDIFF_STEPS,
                    height=config.ANIMATEDIFF_HEIGHT,
                    width=config.ANIMATEDIFF_WIDTH,
                    generator=torch.Generator("cuda").manual_seed(42),
                )

            frames = output.frames[0]

            # Export frames to temp video
            tmp_path = out_path.replace(".mp4", "_animdiff_raw.mp4")
            export_to_video(frames, tmp_path, fps=config.ANIMATEDIFF_FPS_OUT)
            self._log(
                run_id,
                f"AnimateDiff: {len(frames)} frames @ {config.ANIMATEDIFF_FPS_OUT}fps → {tmp_path}",
                "INFO",
            )

            # Optionally unload pipeline to free VRAM for next step
            if config.ANIMATEDIFF_UNLOAD_AFTER_CLIP:
                del pipe
                self._animatediff_pipe = None
                torch.cuda.empty_cache()
                self._log(run_id, "VRAM freed after AnimateDiff generation", "INFO")

            # Upscale from 512x512 to 1920x1080 and loop/trim to duration
            success = self._trim_video_clip(tmp_path, out_path, duration, run_id)
            if success:
                self._log(run_id, f"  ✅ AnimateDiff clip ready: {out_path}", "INFO")
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return success

        except ImportError:
            self._log(run_id, "AnimateDiff: diffusers library not found", "WARNING")
            return False
        except Exception as e:
            # Free VRAM on any error
            try:
                import torch
                self._animatediff_pipe = None
                torch.cuda.empty_cache()
            except Exception:
                pass
            error_str = str(e)
            if "OutOfMemoryError" in error_str or "CUDA out of memory" in error_str:
                self._log(
                    run_id,
                    "CUDA OOM in AnimateDiff — reduce ANIMATEDIFF_NUM_FRAMES in config.py. "
                    "Falling back to next layer.",
                    "WARNING",
                )
            else:
                self._log(run_id, f"AnimateDiff failed: {e}", "WARNING")
            return False

    # ────────────────────────────────────────────────────────────────────
    #  LAYER 3: SVD — animates a static image (optional, slower)
    # ────────────────────────────────────────────────────────────────────
    def _check_svd_available(self, run_id: str) -> bool:
        if not config.ENABLE_SVD_ANIMATION:
            return False
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram < 5.0:
                self._log(run_id, f"SVD: VRAM too low ({vram:.1f}GB)", "WARNING")
                return False
            import diffusers  # noqa
            return True
        except ImportError:
            self._log(run_id, "SVD: diffusers not installed", "WARNING")
            return False
        except Exception:
            return False

    def _load_svd_pipeline(self, run_id: str):
        if self._svd_pipe is not None:
            return self._svd_pipe
        try:
            import torch
            from diffusers import StableVideoDiffusionPipeline

            self._log(run_id, "Loading SVD pipeline (downloads ~7GB on first run)...", "INFO")
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                config.SVD_MODEL_ID, torch_dtype=torch.float16, variant="fp16",
            )
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.unet.enable_forward_chunking()
            self._svd_pipe = pipe
            self._log(run_id, "SVD pipeline loaded (fp16 + CPU offload)", "INFO")
            return self._svd_pipe
        except Exception as e:
            self._log(run_id, f"SVD pipeline load failed: {e}", "WARNING")
            return None

    def _generate_svd_video(
        self, image_path: str, out_path: str, duration: float, run_id: str
    ) -> bool:
        try:
            import torch
            from diffusers.utils import load_image, export_to_video

            pipe = self._load_svd_pipeline(run_id)
            if pipe is None:
                return False

            self._log(run_id, f"SVD animating: {image_path}", "INFO")
            img = load_image(image_path).resize((config.SVD_GEN_WIDTH, config.SVD_GEN_HEIGHT))

            with torch.autocast("cuda", dtype=torch.float16):
                frames = pipe(
                    img,
                    num_frames=config.SVD_NUM_FRAMES,
                    fps_id=config.SVD_FPS,
                    motion_bucket_id=config.SVD_MOTION_BUCKET_ID,
                    decode_chunk_size=config.SVD_DECODE_CHUNK,
                    generator=torch.manual_seed(42),
                ).frames[0]

            tmp_path = out_path.replace(".mp4", "_svd_tmp.mp4")
            export_to_video(frames, tmp_path, fps=config.SVD_FPS)
            success = self._trim_video_clip(tmp_path, out_path, duration, run_id)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            if success:
                self._log(run_id, f"  ✅ SVD clip ready: {out_path}", "INFO")
            return success

        except Exception as e:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            self._log(run_id, f"SVD generation failed: {e}", "WARNING")
            return False

    # ────────────────────────────────────────────────────────────────────
    #  LAYER 4: Pexels Photos (static image, Ken Burns in VideoAgent)
    # ────────────────────────────────────────────────────────────────────
    def _fetch_pexels_photo_url(self, query: str, run_id: str) -> Optional[str]:
        if not config.PEXELS_API_KEY:
            return None
        try:
            resp = requests.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": config.PEXELS_API_KEY},
                params={"query": query, "orientation": "landscape", "size": "large",
                        "per_page": config.PEXELS_PER_SECTION},
                timeout=15,
            )
            if resp.status_code == 429:
                self._log(run_id, "Pexels Photos quota hit", "WARNING")
                return None
            if resp.status_code == 200:
                photos = resp.json().get("photos", [])
                if photos:
                    src = photos[0]["src"]
                    return src.get("large2x") or src.get("large")
        except Exception as e:
            self._log(run_id, f"Pexels Photos error: {e}", "WARNING")
        return None

    # ────────────────────────────────────────────────────────────────────
    #  LAYER 5: Pollinations.ai (free AI image, no key)
    # ────────────────────────────────────────────────────────────────────
    def _pollinations_url(self, query: str) -> str:
        encoded = urllib.parse.quote(query)
        return (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={config.VIDEO_WIDTH}&height={config.VIDEO_HEIGHT}&nologo=true"
        )

    # ────────────────────────────────────────────────────────────────────
    #  Generic helpers
    # ────────────────────────────────────────────────────────────────────
    def _download_image(self, url: str, save_path: str, run_id: str) -> bool:
        try:
            resp = requests.get(url, timeout=60, stream=True)
            if resp.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                return True
        except Exception as e:
            self._log(run_id, f"Image download failed: {e}", "WARNING")
        return False

    # ────────────────────────────────────────────────────────────────────
    #  MAIN EXECUTE
    # ────────────────────────────────────────────────────────────────────
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        run_id         = state["run_id"]
        script_data    = state["script_data"]
        voiceover_data = state["voiceover_data"]
        trend_data     = state["trend_data"]

        topic          = trend_data.get("topic", "technology")
        sections       = script_data.get("sections", [])
        timestamp_map  = voiceover_data.get("timestamp_map", [])

        run_output_dir = os.path.join(config.OUTPUT_DIR, f"run_{run_id}")
        visuals_dir    = os.path.join(run_output_dir, "visuals")
        os.makedirs(visuals_dir, exist_ok=True)

        # Check GPU capabilities once
        use_animatediff = self._check_animatediff_available(run_id)
        use_svd         = self._check_svd_available(run_id)

        self._log(
            run_id,
            f"Visual strategy: AnimateDiff={'ON' if use_animatediff else 'OFF'}, "
            f"SVD={'ON' if use_svd else 'OFF'}",
            "INFO",
        )

        visual_map: List[dict] = []

        for i, section in enumerate(sections):
            title = section.get("title", f"Section {i+1}")
            self._log(run_id, f"[{i+1}/{len(sections)}] Processing: '{title}'", "INFO")

            # Find timing from voiceover timestamp_map
            ts         = next((t for t in timestamp_map if t.get("section_index") == i), None)
            start_time = ts["start_time"] if ts else (i * 30)
            end_time   = ts["end_time"]   if ts else ((i + 1) * 30)
            duration   = max(end_time - start_time, 2.0)

            query = self._generate_visual_query(section, topic, run_id)
            self._log(run_id, f"  Query: '{query}' | Need: {duration:.1f}s", "INFO")

            asset_path = None
            asset_type = "image"

            # ── LAYER 1: Pexels Real Video ────────────────────────────
            pexels_video_url = self._fetch_pexels_video_url(query, duration, run_id)
            if pexels_video_url:
                raw_path     = os.path.join(visuals_dir, f"section_{i:03d}_pexels_raw.mp4")
                trimmed_path = os.path.join(visuals_dir, f"section_{i:03d}_video.mp4")
                if self._download_video(pexels_video_url, raw_path, run_id):
                    if self._trim_video_clip(raw_path, trimmed_path, duration, run_id):
                        asset_path = trimmed_path
                        asset_type = "video"
                        self._log(run_id, f"  ✅ Layer 1: Pexels stock video", "INFO")
                        # Clean up raw download to save space
                        try:
                            os.remove(raw_path)
                        except Exception:
                            pass

            # ── LAYER 2: AnimateDiff (GPU text-to-video) ──────────────
            if asset_path is None and use_animatediff:
                animdiff_path = os.path.join(visuals_dir, f"section_{i:03d}_animdiff.mp4")
                ad_prompt = self._generate_animatediff_prompt(query, section, run_id)
                self._log(run_id, f"  Layer 2: AnimateDiff — '{ad_prompt[:60]}...'", "INFO")
                if self._generate_animatediff_video(ad_prompt, animdiff_path, duration, run_id):
                    asset_path = animdiff_path
                    asset_type = "video"

            # ── LAYER 3: SVD (GPU image animation) ────────────────────
            if asset_path is None and use_svd:
                photo_url = self._fetch_pexels_photo_url(query, run_id)
                if photo_url:
                    photo_path = os.path.join(visuals_dir, f"section_{i:03d}_photo_for_svd.jpg")
                    svd_path   = os.path.join(visuals_dir, f"section_{i:03d}_svd.mp4")
                    if self._download_image(photo_url, photo_path, run_id):
                        self._log(run_id, "  Layer 3: SVD animation...", "INFO")
                        if self._generate_svd_video(photo_path, svd_path, duration, run_id):
                            asset_path = svd_path
                            asset_type = "video"

            # ── LAYER 4: Pexels Photo (static → Ken Burns) ────────────
            if asset_path is None:
                photo_url = self._fetch_pexels_photo_url(query, run_id)
                if photo_url:
                    photo_path = os.path.join(visuals_dir, f"section_{i:03d}_photo.jpg")
                    if self._download_image(photo_url, photo_path, run_id):
                        asset_path = photo_path
                        asset_type = "image"
                        self._log(run_id, "  ✅ Layer 4: Pexels photo (Ken Burns)", "INFO")

            # ── LAYER 5: Pollinations.ai (AI image fallback) ──────────
            if asset_path is None:
                poll_url   = self._pollinations_url(query)
                poll_path  = os.path.join(visuals_dir, f"section_{i:03d}_ai.jpg")
                self._log(run_id, "  Layer 5: Pollinations.ai fallback...", "INFO")
                if self._download_image(poll_url, poll_path, run_id):
                    asset_path = poll_path
                    asset_type = "image"
                else:
                    asset_path = poll_url
                    asset_type = "url"
                    self._log(run_id, "  Using Pollinations URL directly", "WARNING")

            visual_map.append({
                "section_index": i,
                "section_title": title,
                "start_time":    start_time,
                "end_time":      end_time,
                "duration":      duration,
                "asset_path":    asset_path,
                "asset_type":    asset_type,   # "video" | "image" | "url"
                "query_used":    query,
            })

            # Brief pause to respect rate limits
            time.sleep(0.5)

        # Summary
        video_count = sum(1 for v in visual_map if v["asset_type"] == "video")
        image_count = sum(1 for v in visual_map if v["asset_type"] in ("image", "url"))
        self._log(
            run_id,
            f"VisualAgent done: {video_count} real video clips + {image_count} images "
            f"({len(sections)} sections total)",
            "INFO",
        )

        return {
            "visual_data": {
                "visual_map":       visual_map,
                "visuals_dir":      visuals_dir,
                "video_clip_count": video_count,
                "image_count":      image_count,
            }
        }
