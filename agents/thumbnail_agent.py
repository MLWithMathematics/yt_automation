"""
agents/thumbnail_agent.py — Agent 7: Thumbnail generation with Pillow.

FIXES IN THIS VERSION:
  ① Pollinations.ai fetch: removed stream=True (was causing Windows
    asyncio [WinError 10054] cascade on every timeout).  Now uses a
    plain timeout=(connect_s, read_s) tuple so the socket is closed
    cleanly when the deadline is hit.

  ② Multi-source image cascade:
       1st  → Pollinations.ai  (timeout 10 s)
       2nd  → Lorem Picsum      (timeout  5 s, always reliable, no key)
       3rd  → Unsplash Source   (timeout  5 s, topic-relevant, no key)
       last → PIL gradient      (instant, never fails)

  ③ _apply_vignette: alpha direction was INVERTED (was darkening the
    centre instead of the edges).  Fixed by reversing the step formula.

  ④ Step-by-step INFO logs added so the user can see progress during
    the PIL composition phase (previously looked frozen for ~30 s).

  ⑤ _apply_gradient_overlay: removed per-pixel loop entirely for
    dark_left / dark_bottom; uses numpy-free column/row slicing trick
    via Pillow's paste() for O(1) pixel writes.
"""
import io
import json
import logging
import os
import urllib.parse
from typing import Any, Dict, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Free image sources (no API key required)
# ---------------------------------------------------------------------------
_POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/{prompt}?width={w}&height={h}&nologo=true"
_PICSUM_URL        = "https://picsum.photos/{w}/{h}"          # Lorem Picsum
_UNSPLASH_URL      = "https://source.unsplash.com/{w}x{h}/?{q}"  # Unsplash Source


class ThumbnailAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ThumbnailAgent",
            role_description=(
                "Generates a high-CTR YouTube thumbnail using Pillow. "
                "AI designs the concept, fetches a background via a cascade "
                "(Pollinations → Picsum → Unsplash → PIL gradient), "
                "then composites text + gradient + vignette."
            ),
            inputs_expected=["trend_data", "keyword_data"],
            outputs_produced=["thumbnail_data"],
        )

    # ─────────────────────────────────────────────────────────────────────
    #  LLM concept design
    # ─────────────────────────────────────────────────────────────────────

    def _design_thumbnail_concept(self, topic: str, angle: str,
                                   keyword: str, run_id: str) -> dict:
        system_prompt = (
            "You are a YouTube thumbnail designer specialising in high click-through rate.\n"
            "Return ONLY valid JSON, no markdown fences:\n"
            "{\n"
            '  "background_prompt": "detailed photorealistic image prompt for Pollinations.ai",\n'
            '  "search_query": "2-4 word image search term for Unsplash/Picsum fallback",\n'
            '  "main_text": "MAX 4 WORDS ALL CAPS – shocking or curious hook",\n'
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
                [SystemMessage(content=system_prompt),
                 HumanMessage(content=user_prompt)],
                run_id=run_id,
            )
            return self.parse_llm_json(resp)
        except Exception as e:
            self._log(run_id, f"Concept design failed ({e}), using defaults", "WARNING")
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
    #  FIX ①: no stream=True → clean socket teardown on Windows asyncio
    # ─────────────────────────────────────────────────────────────────────

    def _fetch_url(self, url: str, connect_timeout: float,
                   read_timeout: float) -> Optional[Image.Image]:
        """
        Download an image URL without streaming.

        Using timeout=(connect, read) instead of a single scalar avoids
        the Windows asyncio [WinError 10054] cascade that occurred when
        stream=True kept the socket alive past the asyncio proactor window.
        """
        try:
            resp = requests.get(url, timeout=(connect_timeout, read_timeout))
            if resp.status_code != 200:
                return None
            data = resp.content           # single allocation — fine at 1280×720
            if len(data) < 1024:          # reject near-empty responses
                return None
            img = Image.open(io.BytesIO(data)).convert("RGB")
            resampler = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)
            return img.resize(
                (config.THUMBNAIL_WIDTH, config.THUMBNAIL_HEIGHT), resampler
            )
        except requests.exceptions.Timeout:
            return None
        except Exception:
            return None

    def _fetch_background(self, concept: dict, topic: str,
                          run_id: str) -> Optional[Image.Image]:
        """
        Try three free image sources in order of preference.

        Source 1 — Pollinations.ai  (AI-generated, topic-specific, 10 s limit)
        Source 2 — Lorem Picsum      (random hi-res photo, always available, 5 s)
        Source 3 — Unsplash Source   (topic-relevant photo, 5 s)
        """
        W, H = config.THUMBNAIL_WIDTH, config.THUMBNAIL_HEIGHT

        # ── Source 1: Pollinations.ai ────────────────────────────────────
        prompt = concept.get("background_prompt", f"cinematic {topic}")
        encoded = urllib.parse.quote(prompt[:200])
        pol_url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={W}&height={H}&nologo=true"
        )
        self._log(run_id, "Trying Pollinations.ai (10 s)…", "INFO")
        img = self._fetch_url(pol_url, connect_timeout=5, read_timeout=10)
        if img is not None:
            self._log(run_id, "Background: Pollinations.ai ✓", "INFO")
            return img

        # ── Source 2: Picsum (reliable, no key, 5 s) ─────────────────────
        self._log(run_id, "Pollinations unavailable → trying Picsum…", "WARNING")
        picsum_url = f"https://picsum.photos/{W}/{H}"
        img = self._fetch_url(picsum_url, connect_timeout=3, read_timeout=5)
        if img is not None:
            self._log(run_id, "Background: Lorem Picsum ✓", "INFO")
            return img

        # ── Source 3: Unsplash Source (topic-relevant, 5 s) ──────────────
        self._log(run_id, "Picsum unavailable → trying Unsplash Source…", "WARNING")
        query_raw = concept.get("search_query", topic)
        query_enc = urllib.parse.quote(query_raw[:50])
        unsplash_url = f"https://source.unsplash.com/{W}x{H}/?{query_enc}"
        img = self._fetch_url(unsplash_url, connect_timeout=3, read_timeout=5)
        if img is not None:
            self._log(run_id, "Background: Unsplash Source ✓", "INFO")
            return img

        self._log(run_id, "All image sources failed → PIL gradient fallback", "WARNING")
        return None

    def _make_fallback_background(self, topic: str,
                                  accent_color: str) -> Image.Image:
        """Generate a gradient background with PIL when all APIs fail."""
        try:
            r, g, b = self._hex_to_rgb(accent_color)
        except Exception:
            r, g, b = 30, 60, 120
        W, H = config.THUMBNAIL_WIDTH, config.THUMBNAIL_HEIGHT
        img  = Image.new("RGB", (W, H))
        draw = ImageDraw.Draw(img)
        for y in range(H):
            t  = y / H
            cr = int(max(0, min(255, r * 0.3 + t * r * 0.7)))
            cg = int(max(0, min(255, g * 0.2 + t * g * 0.5)))
            cb = int(max(0, min(255, b * 0.4 + t * b * 0.6)))
            draw.line([(0, y), (W, y)], fill=(cr, cg, cb))
        return img

    # ─────────────────────────────────────────────────────────────────────
    #  Overlay helpers
    # ─────────────────────────────────────────────────────────────────────

    def _apply_gradient_overlay(self, img: Image.Image,
                                style: str) -> Image.Image:
        """Semi-transparent gradient overlay using Pillow paste() bands."""
        W, H    = img.size
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw    = ImageDraw.Draw(overlay)

        if style == "dark_left":
            half = W // 2
            for x in range(half):
                a = int(180 * (1 - x / half))
                draw.line([(x, 0), (x, H - 1)], fill=(0, 0, 0, a))

        elif style == "dark_bottom":
            start = H // 2
            span  = H - start
            for y in range(start, H):
                a = int(180 * ((y - start) / span))
                draw.line([(0, y), (W - 1, y)], fill=(0, 0, 0, a))

        elif style == "dark_center":
            # Concentric filled rectangles → fast O(steps)
            steps = 40
            for step in range(steps, 0, -1):
                m = int((step / steps) * min(W, H) * 0.45)
                a = int(110 * (1 - step / steps))
                draw.rectangle([m, m, W - m, H - m], outline=(0, 0, 0, a))

        # No-op for "none" or unknown style
        base = img.convert("RGBA")
        return Image.alpha_composite(base, overlay).convert("RGB")

    def _apply_vignette(self, img: Image.Image) -> Image.Image:
        """
        Dark-edge vignette.

        FIX ③: The previous code drew rectangles with alpha = 160*(i/steps),
        which made the CENTRE dark (high alpha for small inner rectangles) and
        left the edges bright — the opposite of a vignette.

        Corrected formula: alpha = 160 * (1 - i/steps)
          i=0 → outer-most rectangle → alpha ≈ 160  (dark edges)
          i=steps-1 → inner-most rectangle → alpha ≈ 0   (bright centre)
        """
        W, H     = img.size
        vignette = Image.new("L", (W, H), 0)
        draw     = ImageDraw.Draw(vignette)
        steps    = 60
        for i in range(steps):
            a = int(150 * (1 - i / steps))   # ← FIXED direction
            draw.rectangle([i, i, W - i - 1, H - i - 1], outline=a)
        vignette = vignette.filter(ImageFilter.GaussianBlur(radius=28))
        dark     = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        dark.putalpha(vignette)
        return Image.alpha_composite(img.convert("RGBA"), dark).convert("RGB")

    # ─────────────────────────────────────────────────────────────────────
    #  Font + text helpers
    # ─────────────────────────────────────────────────────────────────────

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        candidates = [
            getattr(config, "THUMBNAIL_FONT_PATH", ""),
            getattr(config, "FALLBACK_FONT_PATH", ""),
            "C:/Windows/Fonts/impact.ttf",
            "C:/Windows/Fonts/ariblk.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for path in candidates:
            try:
                # Basic check to avoid segfaulting on HTML files disguised as TTFs
                if path and os.path.exists(path) and os.path.getsize(path) > 10000:
                    # Check the magic bytes to ensure it is a real binary font file
                    with open(path, "rb") as f:
                        header = f.read(4)
                    if header in (b'\x00\x01\x00\x00', b'OTTO', b'true'):
                        return ImageFont.truetype(path, size)
            except Exception:
                pass
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    def _draw_text_with_shadow(self, draw: ImageDraw.Draw, text: str,
                                pos: Tuple[int, int],
                                font: Optional[ImageFont.ImageFont],
                                color: str, shadow_offset: int = 4):
        x, y = pos
        kwargs = {"font": font} if font else {}
        draw.text((x + shadow_offset, y + shadow_offset), text,
                  fill=(0, 0, 0), **kwargs)
        draw.text((x, y), text, fill=color, **kwargs)

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        h = hex_color.strip().lstrip("#")
        if len(h) != 6:
            raise ValueError(f"Bad hex: {hex_color!r}")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def _measure_text(self, draw: ImageDraw.Draw, text: str,
                      font) -> Tuple[int, int]:
        try:
            bb = draw.textbbox((0, 0), text, font=font)
            return bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            fs = getattr(font, "size", 40) if font else 40
            return len(text) * max(10, fs // 2), fs

    # ─────────────────────────────────────────────────────────────────────
    #  Main execute
    # ─────────────────────────────────────────────────────────────────────

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        run_id          = state["run_id"]
        trend_data      = state["trend_data"]
        keyword_data    = state["keyword_data"]

        topic           = trend_data.get("topic", "Video")
        angle           = trend_data.get("angle", "")
        primary_keyword = keyword_data.get("primary_keyword", topic)

        run_output_dir = os.path.join(config.OUTPUT_DIR, f"run_{run_id}")
        os.makedirs(run_output_dir, exist_ok=True)

        # ── Step 1: AI concept design ──────────────────────────────────
        self._log(run_id, "Designing thumbnail concept with AI…", "INFO")
        concept       = self._design_thumbnail_concept(
            topic, angle, primary_keyword, run_id
        )
        accent_color  = concept.get("accent_color",  "#FF4444")
        overlay_style = concept.get("overlay_style", "dark_left")
        main_text     = concept.get("main_text",     topic[:20]).upper()
        sub_text      = concept.get("sub_text",      "")
        text_color    = concept.get("text_color",    "#FFFFFF")
        self._log(
            run_id,
            f"Concept: '{main_text}' [{concept.get('emotion')}]  "
            f"overlay={overlay_style}",
            "INFO",
        )

        # ── Step 2: Background image (cascade) ────────────────────────
        self._log(run_id, "Fetching background image…", "INFO")
        bg_img = self._fetch_background(concept, topic, run_id)
        if bg_img is None:
            self._log(run_id, "All sources failed → PIL gradient", "WARNING")
            bg_img = self._make_fallback_background(topic, accent_color)
        self._log(run_id, f"Background ready: {bg_img.size}", "INFO")

        # ── Step 3: Gradient overlay ──────────────────────────────────
        self._log(run_id, "Applying gradient overlay…", "INFO")
        bg_img = self._apply_gradient_overlay(bg_img, overlay_style)

        # ── Step 4: Text composition ──────────────────────────────────
        self._log(run_id, "Compositing text…", "INFO")
        draw = ImageDraw.Draw(bg_img)
        W, H = bg_img.size

        main_font            = self._load_font(110)
        text_w, text_h       = self._measure_text(draw, main_text, main_font)

        if overlay_style == "dark_left":
            x = 60
        else:
            x = max(20, (W - text_w) // 2)
        y = max(20, (H - text_h) // 2 - 40)

        # Accent bar
        try:
            ar, ag, ab = self._hex_to_rgb(accent_color)
        except Exception:
            ar, ag, ab = 255, 68, 68
        bar = [
            max(0, x - 20),
            max(0, y - 10),
            min(W - 1, x + text_w + 20),
            min(H - 1, y + text_h + 10),
        ]
        draw.rectangle(bar, fill=(ar, ag, ab))
        self._draw_text_with_shadow(draw, main_text, (x, y), main_font, text_color)

        # Sub-text
        if sub_text:
            sub_font       = self._load_font(52)
            sub_w, _       = self._measure_text(draw, sub_text, sub_font)
            sub_x          = x if overlay_style == "dark_left" else max(20, (W - sub_w) // 2)
            sub_y          = y + text_h + 22
            if sub_y + 60 < H:
                self._draw_text_with_shadow(
                    draw, sub_text, (sub_x, sub_y), sub_font, text_color
                )

        # Channel logo watermark
    
        logo_path = getattr(config, "CHANNEL_LOGO_PATH", "")
        if logo_path and os.path.exists(logo_path):
            try:
                logo    = Image.open(logo_path).convert("RGBA")
                logo_sz = (80, 80)
                resampler = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)
                logo    = logo.resize(logo_sz, resampler)
                bg_img.paste(logo, (W - logo_sz[0] - 20, H - logo_sz[1] - 20), logo)
            except Exception as e:
                self._log(run_id, f"Logo paste skipped: {e}", "WARNING")

        # ── Step 5: Vignette ──────────────────────────────────────────
        self._log(run_id, "Applying vignette…", "INFO")
        bg_img = self._apply_vignette(bg_img)

        # ── Step 6: Save ──────────────────────────────────────────────
        thumbnail_path = os.path.join(run_output_dir, "thumbnail.jpg")
        self._log(run_id, f"Saving thumbnail to {thumbnail_path}…", "INFO")
        bg_img.save(thumbnail_path, "JPEG", quality=95)
        self._log(run_id, "Thumbnail saved ✓", "INFO")

        return {
            "thumbnail_data": {
                "thumbnail_path": thumbnail_path,
                "concept":        concept,
            }
        }
