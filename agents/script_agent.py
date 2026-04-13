"""
agents/script_agent.py — Agent 3: Generates retention-optimized video scripts.
"""
import json
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
import config

logger = logging.getLogger(__name__)


class ScriptAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ScriptAgent",
            role_description=(
                "Generates a complete, retention-optimized YouTube video script using "
                "structured storytelling: hook, context, main body with pattern interrupts, "
                "and a value-tied CTA. Script length is calibrated to the configured video length."
            ),
            inputs_expected=["trend_data", "keyword_data"],
            outputs_produced=["script_data"],
        )

    def _get_length_guidance(self) -> dict:
        length_map = {
            "short":  {"minutes": 3,  "sections": 3, "words": 450},
            "medium": {"minutes": 7,  "sections": 5, "words": 1050},
            "long":   {"minutes": 12, "sections": 7, "words": 1800},
        }
        return length_map.get(config.VIDEO_LENGTH, length_map["medium"])

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        run_id = state["run_id"]
        trend_data = state["trend_data"]
        keyword_data = state["keyword_data"]
        niche = state.get("niche", config.NICHE)

        topic = trend_data.get("topic", "")
        angle = trend_data.get("angle", "")
        target_audience = trend_data.get("target_audience", "general audience")
        primary_keyword = keyword_data.get("primary_keyword", topic)
        secondary_keywords = keyword_data.get("secondary_keywords", [])
        search_intent = keyword_data.get("search_intent", "")
        length_guide = self._get_length_guidance()

        self._log(run_id, f"Generating {config.VIDEO_LENGTH} script (~{length_guide['minutes']}min) for: '{topic}'", "INFO")

        system_prompt = f"""You are an elite YouTube scriptwriter specialized in {niche}.
Your scripts are known for exceptional viewer retention and watch time.

SCRIPT STRUCTURE (mandatory):
1. HOOK (0-30s): Open with a bold claim, shocking statistic, or provocative question.
   NEVER start with "Hey guys" or "Welcome back". Jump straight into the tension.
2. CONTEXT (30-60s): Why this matters RIGHT NOW. Tie directly to the current trend.
   Mention primary keyword naturally: "{primary_keyword}"
3. MAIN BODY ({length_guide['sections'] - 2} sections):
   - Each section has a clear title and builds on the previous
   - Insert a pattern interrupt every ~90 seconds: rhetorical question, stat drop,
     or "but here's the thing..." transition
   - Weave in secondary keywords naturally: {secondary_keywords}
4. CTA (final 30s): Specific subscribe reason tied to the video's value proposition.
   NOT generic ("if you liked this video..."). Make it about what they'll miss next.

WRITING RULES:
- No filler words. Every sentence earns its place.
- Write as if speaking: contractions, short punchy sentences, rhetorical questions
- Pattern interrupts keep viewers from clicking away
- Target audience: {target_audience}
- Search intent to satisfy: {search_intent}

OUTPUT FORMAT — return ONLY this JSON structure, no markdown:
{{
  "sections": [
    {{
      "title": "HOOK",
      "content": "full script text for this section",
      "duration_seconds": 30,
      "type": "hook"
    }},
    {{
      "title": "CONTEXT",
      "content": "full script text",
      "duration_seconds": 30,
      "type": "context"
    }},
    ... (main body sections) ...,
    {{
      "title": "CALL TO ACTION",
      "content": "full script text",
      "duration_seconds": 30,
      "type": "cta"
    }}
  ],
  "hook": "the opening line only (first sentence of the hook section)",
  "word_count": estimated_total_word_count,
  "total_duration": total_seconds
}}"""

        user_prompt = f"""Create a complete video script with these parameters:
Topic: {topic}
Angle: {angle}
Target length: ~{length_guide['minutes']} minutes ({length_guide['words']} words)
Primary keyword: {primary_keyword}
Secondary keywords: {secondary_keywords}
Target audience: {target_audience}

Write the FULL script content for every section. Do not use placeholders."""

        self._log(run_id, "Calling Groq LLM for script generation (may take 15-30s)...", "INFO")
        response_text = self.call_llm_with_retry(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
            run_id=run_id,
        )

        try:
            script_data = self.parse_llm_json(response_text)
        except json.JSONDecodeError as e:
            self._log(run_id, f"JSON parse failed: {e}. Attempting text extraction...", "WARNING")
            # Build a minimal script from the raw text response
            target_duration = config.get_target_duration()
            script_data = {
                "sections": [
                    {
                        "title": "HOOK",
                        "content": response_text[:500] if len(response_text) > 500 else response_text,
                        "duration_seconds": 30,
                        "type": "hook",
                    },
                    {
                        "title": "MAIN CONTENT",
                        "content": response_text[500:] if len(response_text) > 500 else f"Today we explore {topic}. {angle}",
                        "duration_seconds": target_duration - 60,
                        "type": "body",
                    },
                    {
                        "title": "CALL TO ACTION",
                        "content": f"Subscribe for more {niche} content. Hit the notification bell so you never miss our next video on {primary_keyword}.",
                        "duration_seconds": 30,
                        "type": "cta",
                    },
                ],
                "hook": f"What you're about to learn about {topic} will change how you think forever.",
                "word_count": len(response_text.split()),
                "total_duration": target_duration,
            }

        # Validate and fix total_duration
        sections = script_data.get("sections", [])
        computed_duration = sum(s.get("duration_seconds", 0) for s in sections)
        if computed_duration == 0:
            for s in sections:
                words = len(s.get("content", "").split())
                s["duration_seconds"] = max(30, int(words / 2.5))  # ~150 WPM
            computed_duration = sum(s.get("duration_seconds", 0) for s in sections)

        script_data["total_duration"] = computed_duration
        script_data["word_count"] = sum(len(s.get("content", "").split()) for s in sections)

        self._log(run_id, f"Script generated: {len(sections)} sections, {script_data['word_count']} words, ~{computed_duration}s", "INFO")
        return {"script_data": script_data}
