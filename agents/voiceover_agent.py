"""
agents/voiceover_agent.py — Agent 4: Text-to-Speech via edge-tts.

FIXES:
  - asyncio.run() replaced with an explicit new-loop approach that works
    correctly when called from uvicorn's thread pool on Windows/Python 3.11.
    asyncio.run() on some Windows builds raises "Cannot run the event loop
    while another loop is running" when uvicorn has already set a global
    event loop policy.
  - Per-run voice override: reads voice from state["config"]["voice"] if
    present, falls back to config.VOICE.
  - Edge-tts network retries improved: 3 retries with 10s gap.
"""
import asyncio
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

import edge_tts

import config
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class VoiceoverAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="VoiceoverAgent",
            role_description=(
                "Converts the video script to natural speech audio using Microsoft edge-tts "
                "neural voices (free). Generates MP3 audio and builds a precise timestamp "
                "map linking each script section to its audio time range."
            ),
            inputs_expected=["script_data", "run_id"],
            outputs_produced=["voiceover_data"],
        )

    def _build_full_text(self, sections: list) -> str:
        parts = []
        for section in sections:
            content = section.get("content", "").strip()
            if content:
                parts.append(content)
        return "\n\n".join(parts)

    def _clean_text_for_tts(self, text: str) -> str:
        """Remove markdown and special chars that confuse TTS engines."""
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*',     r'\1', text)
        text = re.sub(r'#+\s+',           '',    text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'[^\w\s\.,!?\'\-:;()"]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # ─────────────────────────────────────────────────────────────────────
    #  asyncio — safe new-loop runner (works in uvicorn thread pool)
    # ─────────────────────────────────────────────────────────────────────

    def _run_async_in_thread(self, coro) -> Any:
        """
        Run an async coroutine from a synchronous context safely.

        asyncio.run() on Windows + Python 3.11 can fail when uvicorn has
        already configured a ProactorEventLoop policy globally and the current
        thread has a stale reference. Creating an explicit new loop and running
        it to completion is safe from any thread.
        """
        result_holder = [None]
        exc_holder    = [None]

        def runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result_holder[0] = loop.run_until_complete(coro)
            except Exception as e:
                exc_holder[0] = e
            finally:
                loop.close()

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join()

        if exc_holder[0]:
            raise exc_holder[0]
        return result_holder[0]

    # ─────────────────────────────────────────────────────────────────────
    #  edge-tts generation with retry
    # ─────────────────────────────────────────────────────────────────────

    async def _generate_audio_async(self, text: str, output_path: str,
                                     voice: str, run_id: str) -> bool:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(output_path)
                return True
            except Exception as e:
                self._log(run_id,
                          f"edge-tts attempt {attempt+1}/{max_retries} failed: {e}",
                          "WARNING")
                if attempt < max_retries - 1:
                    await asyncio.sleep(10)
        return False

    def _generate_audio(self, text: str, output_path: str,
                        voice: str, run_id: str) -> bool:
        """Generate audio using a fresh asyncio event loop (thread-safe)."""
        coro = self._generate_audio_async(text, output_path, voice, run_id)
        try:
            return self._run_async_in_thread(coro)
        except Exception as e:
            self._log(run_id, f"edge-tts fatal error: {e}", "ERROR")
            return False

    # ─────────────────────────────────────────────────────────────────────
    #  Duration measurement
    # ─────────────────────────────────────────────────────────────────────

    def _get_audio_duration(self, audio_path: str) -> float:
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(audio_path)
            return len(audio) / 1000.0
        except Exception as e:
            logger.warning(f"pydub duration failed ({e}), estimating from file size")
            size = os.path.getsize(audio_path)
            return size / 16000.0     # ~128 kbps MP3 ≈ 16 KB/s

    # ─────────────────────────────────────────────────────────────────────
    #  Timestamp map
    # ─────────────────────────────────────────────────────────────────────

    def _build_timestamp_map(self, sections: list, total_duration: float) -> list:
        """Map each section to audio timestamps proportional to word count."""
        total_words = sum(len(s.get("content", "").split()) for s in sections)
        if total_words == 0:
            total_words = 1

        timestamp_map = []
        current_time  = 0.0
        for i, section in enumerate(sections):
            words      = len(section.get("content", "").split())
            proportion = words / total_words
            duration   = total_duration * proportion

            timestamp_map.append({
                "section_index": i,
                "section_title": section.get("title", f"Section {i}"),
                "section_type":  section.get("type", "body"),
                "start_time":    round(current_time, 2),
                "end_time":      round(current_time + duration, 2),
                "duration":      round(duration, 2),
                "word_count":    words,
            })
            current_time += duration

        return timestamp_map

    # ─────────────────────────────────────────────────────────────────────
    #  Execute
    # ─────────────────────────────────────────────────────────────────────

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        run_id      = state["run_id"]
        script_data = state["script_data"]
        sections    = script_data.get("sections", [])

        if not sections:
            raise ValueError("Script data has no sections")

        # Per-run voice override (set from dashboard Start modal)
        run_cfg = state.get("config", {})
        voice   = run_cfg.get("voice") or config.VOICE

        run_output_dir = os.path.join(config.OUTPUT_DIR, f"run_{run_id}")
        os.makedirs(run_output_dir, exist_ok=True)
        audio_path = os.path.join(run_output_dir, "voiceover.mp3")

        full_text = self._build_full_text(sections)
        full_text = self._clean_text_for_tts(full_text)
        word_count = len(full_text.split())
        self._log(run_id, f"TTS: {word_count} words — voice: {voice}", "INFO")

        self._log(run_id, "Generating audio with edge-tts...", "INFO")
        success = self._generate_audio(full_text, audio_path, voice, run_id)

        if not success or not os.path.exists(audio_path):
            raise RuntimeError("edge-tts failed to generate audio after 3 retries")

        duration = self._get_audio_duration(audio_path)
        self._log(run_id, f"Audio: {audio_path} ({duration:.1f}s)", "INFO")

        timestamp_map = self._build_timestamp_map(sections, duration)
        self._log(run_id, f"Timestamp map: {len(timestamp_map)} sections", "INFO")

        return {
            "voiceover_data": {
                "audio_path":       audio_path,
                "duration_seconds": round(duration, 2),
                "timestamp_map":    timestamp_map,
                "voice_used":       voice,
                "word_count":       word_count,
            }
        }
