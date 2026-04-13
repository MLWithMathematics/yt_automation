"""
agents/metadata_agent.py — Agent 8: YouTube upload metadata generation.

FIXES IN THIS VERSION:
  ① category_id used config.get_category_id() which reads the global
    config.NICHE — not the per-run niche passed in state.  Fixed to use
    config.NICHE_CATEGORY_MAP.get(niche.lower(), config.YOUTUBE_CATEGORY_ID).

  ② POWER_WORDS was rendered as a Python list repr (['Shocking', ...]) in
    the LLM prompt.  Fixed to join as a readable comma-separated string.

  ③ JSON parse fallback now preserves any partial title/description from
    the LLM response instead of silently dropping it.

  ④ tags_ordered character-limit check now counts total chars across all
    tags (YouTube's real limit) instead of checking list length.
"""
import json
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
import config

logger = logging.getLogger(__name__)


class MetadataAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="MetadataAgent",
            role_description=(
                "Generates all YouTube upload metadata: SEO-optimised title, "
                "multi-section description with timestamps, 15 tags ordered by "
                "specificity, hashtags, category, and language settings."
            ),
            inputs_expected=["trend_data", "keyword_data", "script_data"],
            outputs_produced=["metadata"],
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _build_timestamps(self, script_data: dict) -> str:
        """Build YouTube timestamp block from script sections."""
        sections    = script_data.get("sections", [])
        lines       = []
        current_sec = 0
        for section in sections:
            title    = section.get("title", "")
            duration = section.get("duration_seconds", 30)
            mins     = int(current_sec // 60)
            secs     = int(current_sec % 60)
            lines.append(f"{mins:02d}:{secs:02d} {title}")
            current_sec += duration
        return "\n".join(lines) if lines else "00:00 Introduction"

    def _category_for_niche(self, niche: str) -> int:
        """
        FIX ①: Look up the category using the per-run niche variable,
        not config.NICHE (global).
        """
        return config.NICHE_CATEGORY_MAP.get(niche.lower(), config.YOUTUBE_CATEGORY_ID)

    def _truncate_title(self, title: str, max_len: int = 60) -> str:
        """Ensure title fits YouTube's 60-char limit."""
        if len(title) <= max_len:
            return title
        # Try to cut at last word boundary before limit
        cut = title[:max_len].rsplit(" ", 1)[0]
        return (cut + "…") if cut else title[:max_len]

    def _trim_tags(self, tags: list, max_chars: int = 500) -> list:
        """
        FIX ④: YouTube's real constraint is 500 total characters across all
        tags (when joined with commas), not a fixed item count.
        """
        result, total = [], 0
        for tag in tags:
            tag = str(tag).strip()
            cost = len(tag) + (1 if result else 0)   # +1 for comma separator
            if total + cost > max_chars:
                break
            result.append(tag)
            total += cost
        return result

    # ─────────────────────────────────────────────────────────────────────
    #  Main execute
    # ─────────────────────────────────────────────────────────────────────

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        run_id      = state["run_id"]
        trend_data  = state["trend_data"]
        kw_data     = state["keyword_data"]
        script_data = state["script_data"]
        niche       = state.get("niche", config.NICHE)

        topic              = trend_data.get("topic", "")
        angle              = trend_data.get("angle", "")
        target_audience    = trend_data.get("target_audience", "general audience")
        primary_keyword    = kw_data.get("primary_keyword", topic)
        secondary_keywords = kw_data.get("secondary_keywords", [])
        tags               = kw_data.get("tags", [])
        hashtags           = kw_data.get("hashtags", [])
        hook               = script_data.get("hook", "")
        timestamps_str     = self._build_timestamps(script_data)
        total_duration     = script_data.get("total_duration", 420)
        mins               = int(total_duration // 60)
        category_id        = self._category_for_niche(niche)   # FIX ①

        # FIX ②: join POWER_WORDS as a readable string for the LLM
        power_words_str = ", ".join(config.POWER_WORDS)

        self._log(run_id, f"Generating YouTube metadata for: '{topic}'", "INFO")

        system_prompt = f"""You are a YouTube SEO expert specialising in the '{niche}' niche.
Generate complete YouTube upload metadata that maximises search visibility and CTR.

OUTPUT — return ONLY valid JSON with NO markdown fences:
{{
  "title": "Front-loaded with primary keyword, max 60 chars, includes one power word",
  "description": "Full multi-line description as specified below",
  "category_id": {category_id},
  "default_language": "en",
  "tags_ordered": ["most_specific_tag", "...", "broad_niche_tag"]
}}

TITLE RULES:
- Primary keyword must appear in the first 40 characters
- Must include ONE power word from: {power_words_str}
- Maximum 60 characters STRICTLY — count carefully
- No clickbait that misrepresents the content

DESCRIPTION STRUCTURE (write ALL sections):
Lines 1-2  : Hook restatement with primary keyword — creates curiosity above the fold.
Lines 3-10 : Expanded content summary weaving in secondary keywords naturally:
             {', '.join(secondary_keywords[:6]) if secondary_keywords else 'none'}
Line 11    : Blank line, then "⏱ TIMESTAMPS" header + the timestamps list.
Lines 12-14: CTA — subscribe prompt with specific value proposition + social placeholders.
Lines 15-18: SEO paragraph — naturally worded sentences using remaining keywords.
Lines 19-20: Credits / disclaimer if applicable (can be short).

TAGS RULES:
- Reorder the provided tags from most specific → most broad.
- Keep total tag character count under 500.
"""

        user_prompt = f"""Topic: {topic}
Angle: {angle}
Target audience: {target_audience}
Primary keyword: {primary_keyword}
Secondary keywords: {', '.join(secondary_keywords) if secondary_keywords else 'none'}
Hook: {hook}
Video length: ~{mins} minutes
Category ID: {category_id}

Timestamps to embed verbatim:
{timestamps_str}

Available hashtags: {', '.join(hashtags) if hashtags else 'none'}
Available tags: {tags}

Generate the complete metadata package now."""

        response_text = self.call_llm_with_retry(
            [SystemMessage(content=system_prompt),
             HumanMessage(content=user_prompt)],
            run_id=run_id,
        )

        # ── Parse LLM JSON ──────────────────────────────────────────────
        raw = {}
        try:
            raw = self.parse_llm_json(response_text)
        except (json.JSONDecodeError, ValueError) as e:
            self._log(run_id,
                      f"JSON parse failed ({e}) — using fallback metadata",
                      "WARNING")
            # FIX ③: attempt to extract a partial title from raw text
            for line in response_text.splitlines():
                line = line.strip()
                if line.startswith('"title"') or line.lower().startswith("title:"):
                    raw["title"] = line.split(":", 1)[-1].strip().strip('"').strip(",")
                    break

        # ── Build final metadata with fallbacks ────────────────────────
        raw_title = raw.get("title", f"The Truth About {primary_keyword}")
        title     = self._truncate_title(raw_title)

        description = raw.get("description") or (
            f"{hook}\n\n"
            f"In this video we explore {topic} — {angle}\n\n"
            f"Perfect for {target_audience} who want to understand "
            f"{primary_keyword}"
            + (f" and {', '.join(secondary_keywords[:3])}" if secondary_keywords else "")
            + f".\n\n"
            f"⏱ TIMESTAMPS:\n{timestamps_str}\n\n"
            f"Subscribe for more {niche} content every week! 🔔\n\n"
            + (f"Keywords: {', '.join(secondary_keywords)}" if secondary_keywords else "")
        )

        # Append top 3 hashtags below description
        top_hashtags = hashtags[:3] if hashtags else []
        if top_hashtags:
            description = description.rstrip() + "\n\n" + " ".join(top_hashtags)

        final_tags = raw.get("tags_ordered", tags)
        if not isinstance(final_tags, list):
            final_tags = tags
        final_tags = self._trim_tags(final_tags)   # FIX ④

        metadata = {
            "title":            title,
            "description":      description,
            "tags":             final_tags,
            "hashtags":         top_hashtags,
            "category_id":      raw.get("category_id", category_id),
            "default_language": raw.get("default_language", "en"),
            "topic":            topic,
            "primary_keyword":  primary_keyword,
        }

        self._log(run_id, f"Metadata ready ✓  Title: '{title}'", "INFO")
        return {"metadata": metadata}
