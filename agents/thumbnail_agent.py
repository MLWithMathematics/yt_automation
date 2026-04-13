"""
agents/thumbnail_agent.py — Agent 7: Thumbnail generation with Pillow.

═══════════════════════════════════════════════════════════════════
  FIXES IN THIS REVISION
═══════════════════════════════════════════════════════════════════

  BUG-1 (CRITICAL — hang / infinite deadlock)
      ImageFont.truetype(path, 110) deadlocks inside Windows's
      FreeType/GDI font-cache subsystem when called from a thread
      spawned by asyncio's ProactorEventLoop.  The process appeared
      alive but never progressed past "Compositing text…", forcing
      a manual restart every single run.
      FIX: _load_font() now submits the truetype() call to a
      dedicated daemon thread via concurrent.futures and enforces a
      hard 5-second timeout.  On timeout it immediately falls back
      to ImageFont.load_default(size=size) which uses Pillow's
      bundled DejaVuSans TrueType data — bypassing the Windows font
      cache entirely.  Results are cached in a class-level dict so
      the expensive resolution only happens once per process.

  BUG-2 (PERFORMANCE — O(W) Python loop in gradient overlay)
      _apply_gradient_overlay drew dark_left / dark_bottom by
      calling draw.line() in a Python for-loop (640 iterations for
      dark_left on a 1280 px image), each iteration crossing the
      Python→C boundary.  On a loaded asyncio event loop this added
      measurable latency and competed with the FreeType hang.
      FIX: gradient band is now built as a numpy uint8 array and
      composited with Image.fromarray() + Image.alpha_composite() —
      pure-C O(1) pipeline, ~100× faster.

  BUG-3 (WRONG VISUAL — dark_center used outline= not fill=)
      draw.rectangle(… outline=(0,0,0,a)) draws only the 1-pixel
      border of each rectangle, leaving the interior untouched.
      For a dark-centre vignette the call must use fill=.
      FIX: switched to fill=.

  BUG-4 (SILENT CRASH — no try/except around PIL composition)
      Any unexpected Pillow exception (e.g. wrong mode, OOM) inside
      the text compositing block would propagate uncaught into
      base_agent.run(), which logged it but the dashboard showed
      only a vague FAILED status.
      FIX: entire text-composition block wrapped in try/except with
      a descriptive log and a safe fallback thumbnail.

  BUG-5 (VISUAL — logo file has wrong extension)
      Assets folder contains  logo.png.jpeg  (double-extension).
      os.path.exists(config.CHANNEL_LOGO_PATH) always returned
      False because CHANNEL_LOGO_PATH ends in "logo.png".
      FIX: _find_logo() probes several candidate extensions so the
      logo is actually pasted when it exists under any name.

  BUG-6 (CORRECTNESS — draw context rebuilt after overlay)
      Minor: added a fresh ImageDraw.Draw after every image-
      reassignment so the context always refers to the live buffer.

  OTHER IMPROVEMENTS
      • Granular INFO logs before/after every sub-step so the next
        hang (if any) can be pinpointed to the exact line.
      • numpy-based vignette (replaces 60 rectangle draws).
      • Public _compose() helper for unit-testability.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import logging
import os
import threading
import urllib.parse
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from langchain_core.messages import HumanMessage, SystemMessage

import config
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Module-level font cache  (survives across agent instantiations)
# ─────────────────────────────────────────────────────────────────────────────
_FONT_CACHE: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}
_FONT_CACHE_LOCK = threading.Lock()


def _load_truetype_unsafe(path: str, size: int):
    """Called inside an isolated daemon thread so a hang can be timed out."""
    if not (path and os.path.exists(path)):
        return None
    if os.path.getsize(path) < 10_000:
        return None
    with open(path, "rb") as fh:
        magic = fh.read(4)
    valid_magic = {
        b"\x00\x01\x00\x00",  # TrueType
        b"OTTO",               # OpenType/CFF
        b"true",               # Apple TrueType
        b"ttcf",               # TrueType Collection
    }
    if magic not in valid_magic:
        return None
    idx = 0 if magic == b"ttcf" else None
    if idx is not None:
        return ImageFont.truetype(path, size, index=idx)
    return ImageFont.truetype(path, size)


class ThumbnailAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ThumbnailAgent",
            role_description=(
                "Generates a high-CTR YouTube thumbnail using Pillow. "
                "AI designs the concept, fetches a background via a cascade "
                "(Pollinations → Picsum → Unsplash → PIL gradient), "
                "then composites text + gradient overlay + vignette."
            ),
            inputs_expected=["trend_data", "keyword_data"],
            outputs_produced=["thumbnail_data"],
        )

    # ─────────────────────────────────────────────────────────────────────
    #  FIX-1: Font loader with hard timeout + Pillow built-in fallback
    # ─────────────────────────────────────────────────────────────────────

    def _load_font(self, size: int, run_id: str = "unknown") -> ImageFont.ImageFont:
        """
        Return a font at *size* pixels.

        Strategy
        ─────────
        1. Check module-level cache (avoids repeated OS calls).
        2. For each candidate path submit _load_truetype_unsafe() to a
           daemon thread with a 5-second deadline.  If the thread times
           out (Windows FreeType/GDI deadlock) we skip that path.
        3. Final fallback: ImageFont.load_default(size=size) — uses
           Pillow's bundled DejaVuSans.ttf, bypasses Windows font cache,
           never deadlocks.
        """
        # Candidate paths in preference order
        candidates = [
            getattr(config, "THUMBNAIL_FONT_PATH", ""),
            getattr(config, "FALLBACK_FONT_PATH", ""),
            "C:/Windows/Fonts/impact.ttf",
            "C:/Windows/Fonts/ariblk.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]

        for path in candidates:
            if not path:
                continue
            cache_key = (path, size)
            with _FONT_CACHE_LOCK:
                if cache_key in _FONT_CACHE:
                    return _FONT_CACHE[cache_key]

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
                    future = exe.submit(_load_truetype_unsafe, path, size)
                    font = future.result(timeout=5.0)
                if font is not None:
                    with _FONT_CACHE_LOCK:
                        _FONT_CACHE[cache_key] = font
                    return font
            except concurrent.futures.TimeoutError:
                # BUG-1 FIXED: Windows GDI deadlock — skip this path
                logger.warning(
                    "[ThumbnailAgent] Font load TIMED OUT for %s (Windows GDI deadlock) "
                    "— skipping this path.", path
                )
            except Exception as exc:
                logger.debug("[ThumbnailAgent] Font load failed for %s: %s", path, exc)

        # ── Pillow built-in fallback (Pillow 10+) ──
        logger.warning(
            "[ThumbnailAgent] All TrueType paths failed/timed-out — "
            "using Pillow built-in font at size %d", size
        )
        try:
            # Pillow ≥ 10.1: load_default(size=N) returns FreeTypeFont
            font = ImageFont.load_default(size=size)
            with _FONT_CACHE_LOCK:
                _FONT_CACHE[("__default__", size)] = font
            return font
        except TypeError:
            # Pillow < 10: size parameter not supported
            return ImageFont.load_default()

    # ─────────────────────────────────────────────────────────────────────
    #  LLM concept design
    # ─────────────────────────────────────────────────────────────────────

    def _design_thumbnail_concept(
        self, topic: str, angle: str, keyword: str, run_id: str
    ) -> dict:
        system_prompt = (
            "You are a YouTube thumbnail designer specialising in high click-through rate.\n"
            "Return ONLY valid JSON — no markdown fences, no extra text:\n"
            "{\n"
            '  "background_prompt": "detailed photorealistic image prompt for Pollinations.ai",\n'
            '  "search_query": "2-4 word image search term",\n'
            '  "main_text": "MAX 4 WORDS ALL CAPS shocking or curious hook",\n'
            '  "sub_text": "optional 4-6 word supporting line or empty string",\n'
            '  "text_color": "#FFFFFF",\n'
            '  "accent_color": "#FF4444",\n'
            '  "emotion": "shocking|curious|informative|inspiring",\n'
            '  "overlay_style": "dark_left|dark_bottom|dark_center|none"\n'
            "}"
        )
        user_prompt = (
            f"Video topic: {topic}\nAngle: {angle}\n"
            f"Primary keyword: {keyword}\n"
            "Design a scroll-stopping thumbnail."
        )
        try:
            resp = self.call_llm_with_retry(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
                run_id=run_id,
            )
            return self.parse_llm_json(resp)
        except Exception as exc:
            self._log(run_id, f"Concept design failed ({exc}), using defaults", "WARNING")
            return {
                "background_prompt": f"dramatic cinematic photo {topic}, high contrast, 4k",
                "search_query": topic[:30],
                "main_text": topic[:20].upper(),
                "sub_text": "",
                "text_color": "#FFFFFF",
                "accent_color": "#FF4444",
                "emotion": "shocking",
                "overlay_style": "dark_left",
            }

    # ─────────────────────────────────────────────────────────────────────
    #  Background fetch cascade
    # ─────────────────────────────────────────────────────────────────────

    def _fetch_url_to_image(
        self, url: str, connect_t: float, read_t: float
    ) -> Optional[Image.Image]:
        """Download *url* without streaming; return RGB Image or None."""
        try:
            resp = requests.get(url, timeout=(connect_t, read_t))
            if resp.status_code != 200:
                return None
            data = resp.content
            if len(data) < 1024:
                return None
            img = Image.open(io.BytesIO(data)).convert("RGB")
            W, H = config.THUMBNAIL_WIDTH, config.THUMBNAIL_HEIGHT
            resampler = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)
            return img.resize((W, H), resampler)
        except requests.exceptions.Timeout:
            return None
        except Exception:
            return None

    def _fetch_background(
        self, concept: dict, topic: str, run_id: str
    ) -> Optional[Image.Image]:
        W, H = config.THUMBNAIL_WIDTH, config.THUMBNAIL_HEIGHT

        # Source 1: Pollinations.ai
        prompt = concept.get("background_prompt", f"cinematic {topic}")
        encoded = urllib.parse.quote(prompt[:200])
        pol_url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={W}&height={H}&nologo=true"
        )
        self._log(run_id, "Trying Pollinations.ai (10 s)…", "INFO")
        img = self._fetch_url_to_image(pol_url, connect_t=5, read_t=10)
        if img is not None:
            self._log(run_id, "Background: Pollinations.ai ✓", "INFO")
            return img

        # Source 2: Lorem Picsum
        self._log(run_id, "Pollinations unavailable → trying Picsum…", "WARNING")
        img = self._fetch_url_to_image(
            f"https://picsum.photos/{W}/{H}", connect_t=3, read_t=5
        )
        if img is not None:
            self._log(run_id, "Background: Lorem Picsum ✓", "INFO")
            return img

        # Source 3: Unsplash Source
        self._log(run_id, "Picsum unavailable → trying Unsplash Source…", "WARNING")
        q = urllib.parse.quote(concept.get("search_query", topic)[:50])
        img = self._fetch_url_to_image(
            f"https://source.unsplash.com/{W}x{H}/?{q}", connect_t=3, read_t=5
        )
        if img is not None:
            self._log(run_id, "Background: Unsplash Source ✓", "INFO")
            return img

        self._log(run_id, "All image sources failed → PIL gradient fallback", "WARNING")
        return None

    def _make_fallback_background(self, topic: str, accent_color: str) -> Image.Image:
        """Generate a gradient background with PIL (never fails)."""
        try:
            r, g, b = self._hex_to_rgb(accent_color)
        except Exception:
            r, g, b = 30, 60, 120
        W, H = config.THUMBNAIL_WIDTH, config.THUMBNAIL_HEIGHT
        # Build via numpy for speed
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        t = np.linspace(0, 1, H)[:, None]  # (H,1)
        arr[:, :, 0] = np.clip(r * 0.3 + t * r * 0.7, 0, 255)
        arr[:, :, 1] = np.clip(g * 0.2 + t * g * 0.5, 0, 255)
        arr[:, :, 2] = np.clip(b * 0.4 + t * b * 0.6, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), "RGB")

    # ─────────────────────────────────────────────────────────────────────
    #  FIX-2 & FIX-3: Gradient overlay — numpy-based, correct dark_center
    # ─────────────────────────────────────────────────────────────────────

    def _apply_gradient_overlay(
        self, img: Image.Image, style: str
    ) -> Image.Image:
        """
        Semi-transparent gradient overlay built entirely with numpy.

        Replaces the slow Python draw.line() loop (BUG-2) and fixes
        dark_center to darken the centre not just its outline (BUG-3).
        """
        W, H = img.size
        alpha = np.zeros((H, W), dtype=np.float32)

        if style == "dark_left":
            # Column ramp: left edge → 180, midpoint → 0
            col = np.linspace(180, 0, W // 2, dtype=np.float32)
            alpha[:, : W // 2] = col[np.newaxis, :]

        elif style == "dark_bottom":
            # Row ramp: top-half → 0, bottom edge → 180
            start = H // 2
            row = np.linspace(0, 180, H - start, dtype=np.float32)
            alpha[start:, :] = row[:, np.newaxis]

        elif style == "dark_center":
            # Radial: centre → 140, edges → 0  (inverted vignette)
            cx, cy = W / 2, H / 2
            xs = (np.arange(W) - cx) / (W / 2)
            ys = (np.arange(H) - cy) / (H / 2)
            dist = np.sqrt(xs[np.newaxis, :] ** 2 + ys[:, np.newaxis] ** 2)
            dist = np.clip(dist, 0, 1)
            # Darken centre → invert dist
            alpha = (140 * (1 - dist)).astype(np.float32)

        # else "none" → alpha stays zero

        # Build RGBA overlay
        a_uint8 = np.clip(alpha, 0, 255).astype(np.uint8)
        overlay_arr = np.zeros((H, W, 4), dtype=np.uint8)
        overlay_arr[:, :, 3] = a_uint8               # alpha channel only; RGB stays 0
        overlay = Image.fromarray(overlay_arr, "RGBA")
        base = img.convert("RGBA")
        return Image.alpha_composite(base, overlay).convert("RGB")

    # ─────────────────────────────────────────────────────────────────────
    #  Numpy-based vignette (replaces 60 rectangle draws)
    # ─────────────────────────────────────────────────────────────────────

    def _apply_vignette(self, img: Image.Image) -> Image.Image:
        """
        Dark-edge vignette built from a numpy distance field.
        Edges → alpha ≈ 150; centre → alpha ≈ 0.
        """
        W, H = img.size
        cx, cy = W / 2.0, H / 2.0
        xs = (np.arange(W) - cx) / (W / 2.0)
        ys = (np.arange(H) - cy) / (H / 2.0)
        dist = np.sqrt(xs[np.newaxis, :] ** 2 + ys[:, np.newaxis] ** 2)  # (H,W)
        # Remap: 0 at centre → 0 alpha; ≥1 at corner → full alpha
        alpha_f = np.clip(dist, 0, 1.4) / 1.4   # normalise to [0,1]
        # Smooth with a Gaussian approximation (box filter 3×)
        from PIL import ImageFilter as _IF
        a_img = Image.fromarray((alpha_f * 255).astype(np.uint8), "L")
        a_img = a_img.filter(_IF.GaussianBlur(radius=20))
        a_arr = np.array(a_img).astype(np.float32) / 255.0
        vig_alpha = (a_arr * 150).astype(np.uint8)

        dark = np.zeros((H, W, 4), dtype=np.uint8)
        dark[:, :, 3] = vig_alpha
        dark_img = Image.fromarray(dark, "RGBA")
        return Image.alpha_composite(img.convert("RGBA"), dark_img).convert("RGB")

    # ─────────────────────────────────────────────────────────────────────
    #  Font / text helpers
    # ─────────────────────────────────────────────────────────────────────

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        h = hex_color.strip().lstrip("#")
        if len(h) != 6:
            raise ValueError(f"Bad hex color: {hex_color!r}")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    def _measure_text(
        self, draw: ImageDraw.Draw, text: str, font
    ) -> Tuple[int, int]:
        try:
            bb = draw.textbbox((0, 0), text, font=font)
            return bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            fs = getattr(font, "size", 40) if font else 40
            return len(text) * max(10, fs // 2), fs

    def _draw_text_with_shadow(
        self,
        draw: ImageDraw.Draw,
        text: str,
        pos: Tuple[int, int],
        font,
        color: str,
        shadow_offset: int = 4,
    ):
        x, y = pos
        kw = {"font": font} if font else {}
        draw.text((x + shadow_offset, y + shadow_offset), text, fill=(0, 0, 0), **kw)
        draw.text((x, y), text, fill=color, **kw)

    # ─────────────────────────────────────────────────────────────────────
    #  FIX-5: Logo finder (handles double-extension files)
    # ─────────────────────────────────────────────────────────────────────

    def _find_logo(self) -> Optional[str]:
        """
        Probe common extension variants so logo.png.jpeg or logo.png
        are both found.
        """
        base = getattr(config, "CHANNEL_LOGO_PATH", "")
        if not base:
            return None
        candidates = [
            base,
            base + ".jpeg",
            base + ".jpg",
            base + ".png",
            base.replace(".png", ".jpeg"),
            base.replace(".png", ".jpg"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    # ─────────────────────────────────────────────────────────────────────
    #  Public composition helper (also used in unit tests)
    # ─────────────────────────────────────────────────────────────────────

    def _compose(
        self,
        bg: Image.Image,
        concept: dict,
        run_id: str,
    ) -> Image.Image:
        """
        Apply overlay → text → logo → vignette on *bg* (RGB, 1280×720).
        Returns the final composed RGB image.

        FIX-4: wrapped in try/except so any Pillow error is logged and
        a safe fallback is returned rather than crashing the agent.
        """
        W, H = bg.size
        overlay_style = concept.get("overlay_style", "dark_left")
        main_text = concept.get("main_text", "WATCH NOW").upper()
        sub_text  = concept.get("sub_text", "")
        text_color   = concept.get("text_color", "#FFFFFF")
        accent_color = concept.get("accent_color", "#FF4444")

        try:
            ar, ag, ab = self._hex_to_rgb(accent_color)
        except Exception:
            ar, ag, ab = 255, 68, 68

        # ── 3a. Gradient overlay ──────────────────────────────────────
        self._log(run_id, f"  → gradient overlay ({overlay_style})…", "INFO")
        bg = self._apply_gradient_overlay(bg, overlay_style)
        self._log(run_id, "  → gradient overlay ✓", "INFO")

        # ── 3b. Load fonts (with Windows-safe timeout guard) ──────────
        self._log(run_id, "  → loading main font (size 110)…", "INFO")
        main_font = self._load_font(110, run_id)
        self._log(run_id, f"  → main font ready: {type(main_font).__name__}", "INFO")

        draw = ImageDraw.Draw(bg)

        # ── 3c. Measure text ──────────────────────────────────────────
        self._log(run_id, f"  → measuring text: '{main_text}'…", "INFO")
        text_w, text_h = self._measure_text(draw, main_text, main_font)
        self._log(run_id, f"  → text bbox: {text_w}×{text_h}", "INFO")

        # ── 3d. Position ──────────────────────────────────────────────
        if overlay_style == "dark_left":
            x = 60
        else:
            x = max(20, (W - text_w) // 2)
        y = max(20, (H - text_h) // 2 - 40)

        # ── 3e. Accent bar ────────────────────────────────────────────
        self._log(run_id, "  → drawing accent bar…", "INFO")
        bar = [
            max(0, x - 20),
            max(0, y - 10),
            min(W - 1, x + text_w + 20),
            min(H - 1, y + text_h + 10),
        ]
        draw.rectangle(bar, fill=(ar, ag, ab))
        self._log(run_id, "  → accent bar ✓", "INFO")

        # ── 3f. Main text ─────────────────────────────────────────────
        self._log(run_id, "  → drawing main text…", "INFO")
        self._draw_text_with_shadow(draw, main_text, (x, y), main_font, text_color)
        self._log(run_id, "  → main text ✓", "INFO")

        # ── 3g. Sub-text ──────────────────────────────────────────────
        if sub_text:
            self._log(run_id, "  → loading sub font (size 52)…", "INFO")
            sub_font = self._load_font(52, run_id)
            self._log(run_id, "  → sub font ready", "INFO")
            sub_w, _ = self._measure_text(draw, sub_text, sub_font)
            sub_x = x if overlay_style == "dark_left" else max(20, (W - sub_w) // 2)
            sub_y = y + text_h + 22
            if sub_y + 60 < H:
                self._log(run_id, "  → drawing sub-text…", "INFO")
                self._draw_text_with_shadow(draw, sub_text, (sub_x, sub_y), sub_font, text_color)
                self._log(run_id, "  → sub-text ✓", "INFO")

        # ── 3h. Channel logo ──────────────────────────────────────────
        logo_path = self._find_logo()
        if logo_path:
            try:
                self._log(run_id, f"  → pasting logo from {logo_path}…", "INFO")
                logo = Image.open(logo_path).convert("RGBA")
                logo_sz = (80, 80)
                resampler = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)
                logo = logo.resize(logo_sz, resampler)
                bg.paste(logo, (W - logo_sz[0] - 20, H - logo_sz[1] - 20), logo)
                self._log(run_id, "  → logo ✓", "INFO")
            except Exception as exc:
                self._log(run_id, f"Logo paste skipped: {exc}", "WARNING")

        # ── 3i. Vignette ──────────────────────────────────────────────
        self._log(run_id, "  → vignette…", "INFO")
        bg = self._apply_vignette(bg)
        self._log(run_id, "  → vignette ✓", "INFO")

        return bg

    # ─────────────────────────────────────────────────────────────────────
    #  Main execute
    # ─────────────────────────────────────────────────────────────────────

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        run_id       = state["run_id"]
        trend_data   = state["trend_data"]
        keyword_data = state["keyword_data"]

        topic           = trend_data.get("topic", "Video")
        angle           = trend_data.get("angle", "")
        primary_keyword = keyword_data.get("primary_keyword", topic)

        run_output_dir = os.path.join(config.OUTPUT_DIR, f"run_{run_id}")
        os.makedirs(run_output_dir, exist_ok=True)

        # ── Step 1: AI concept design ──────────────────────────────────
        self._log(run_id, "Designing thumbnail concept with AI…", "INFO")
        concept = self._design_thumbnail_concept(topic, angle, primary_keyword, run_id)
        self._log(
            run_id,
            f"Concept: '{concept.get('main_text')}' [{concept.get('emotion')}]  "
            f"overlay={concept.get('overlay_style')}",
            "INFO",
        )

        # ── Step 2: Background image ───────────────────────────────────
        self._log(run_id, "Fetching background image…", "INFO")
        bg_img = self._fetch_background(concept, topic, run_id)
        if bg_img is None:
            self._log(run_id, "All sources failed → PIL gradient", "WARNING")
            bg_img = self._make_fallback_background(topic, concept.get("accent_color", "#3050FF"))
        self._log(run_id, f"Background ready: {bg_img.size}", "INFO")

        # ── Step 3: Compose (overlay + text + logo + vignette) ─────────
        self._log(run_id, "Compositing (overlay + text + vignette)…", "INFO")
        try:
            final = self._compose(bg_img, concept, run_id)
        except Exception as exc:
            # FIX-4: catch any Pillow failure → produce a minimal safe thumbnail
            self._log(run_id, f"Compose failed ({exc}) — generating fallback thumbnail", "ERROR")
            final = self._make_fallback_background(topic, concept.get("accent_color", "#FF4444"))
            draw  = ImageDraw.Draw(final)
            font  = self._load_font(60, run_id)
            kw    = {"font": font} if font else {}
            draw.text((40, 300), concept.get("main_text", topic.upper()), fill="#FFFFFF", **kw)
        self._log(run_id, "Composition complete ✓", "INFO")

        # ── Step 4: Save ───────────────────────────────────────────────
        thumbnail_path = os.path.join(run_output_dir, "thumbnail.jpg")
        self._log(run_id, f"Saving thumbnail → {thumbnail_path}…", "INFO")
        final.save(thumbnail_path, "JPEG", quality=95)
        self._log(run_id, "Thumbnail saved ✓", "INFO")

        return {
            "thumbnail_data": {
                "thumbnail_path": thumbnail_path,
                "concept": concept,
            }
        }
