"""
agents/keyword_agent.py — Agent 2: SEO keyword research.
"""
import json
import requests
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
import config

logger = logging.getLogger(__name__)


class KeywordAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="KeywordAgent",
            role_description=(
                "Performs SEO keyword research for the chosen video topic by scraping "
                "YouTube autocomplete suggestions, then uses AI to cluster keywords by "
                "intent and produce a complete SEO package: primary keyword, secondary "
                "keywords, tags, and hashtags."
            ),
            inputs_expected=["trend_data"],
            outputs_produced=["keyword_data"],
        )

    def _fetch_youtube_autocomplete(self, query: str, run_id: str) -> list:
        """Scrape YouTube autocomplete suggestions."""
        suggestions = []
        try:
            url = "https://suggestqueries.google.com/complete/search"
            params = {"client": "youtube", "ds": "yt", "q": query, "hl": "en"}
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                # Response is JSONP: window.google.ac.h([...])
                text = resp.text
                # Extract JSON array from response
                start = text.index("[")
                data = json.loads(text[start:text.rindex("]") + 1])
                if isinstance(data, list) and len(data) > 1:
                    raw_suggestions = data[1]
                    for item in raw_suggestions:
                        if isinstance(item, list) and item:
                            suggestions.append(item[0])
                        elif isinstance(item, str):
                            suggestions.append(item)
        except Exception as e:
            self._log(run_id, f"Autocomplete fetch error: {e}", "WARNING")
        return suggestions[:20]

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        run_id = state["run_id"]
        trend_data = state["trend_data"]
        topic = trend_data.get("topic", "")
        niche = state.get("niche", config.NICHE)

        self._log(run_id, f"Fetching YouTube autocomplete for: '{topic}'", "INFO")
        suggestions = self._fetch_youtube_autocomplete(topic, run_id)

        # Also fetch for niche-level suggestions
        niche_suggestions = self._fetch_youtube_autocomplete(niche, run_id)
        all_suggestions = list(set(suggestions + niche_suggestions))
        self._log(run_id, f"Collected {len(all_suggestions)} unique autocomplete suggestions", "INFO")

        system_prompt = f"""You are an expert YouTube SEO strategist for the {niche} niche.
Analyze the keyword suggestions and produce a complete SEO package.

Return ONLY valid JSON (no markdown, no extra text):
{{
  "primary_keyword": "main keyword for title (specific, high-intent)",
  "secondary_keywords": ["kw1","kw2","kw3","kw4","kw5"],
  "tags": ["tag1","tag2",...],
  "hashtags": ["#hashtag1","#hashtag2","#hashtag3","#hashtag4","#hashtag5"],
  "search_intent": "what searchers want when they search this topic",
  "keyword_clusters": {{
    "informational": ["kw1","kw2"],
    "how_to": ["kw3","kw4"],
    "comparison": ["kw5"]
  }}
}}

Rules:
- primary_keyword: front-load the most searched phrase, max 60 chars
- secondary_keywords: exactly 5, weave into description naturally
- tags: exactly 15, ordered from most specific to most broad
- hashtags: exactly 5, mix trending + niche-specific"""

        user_prompt = f"""Video topic: {topic}
Target audience: {trend_data.get('target_audience', '')}
Niche: {niche}
YouTube autocomplete suggestions: {json.dumps(all_suggestions)}

Generate the complete SEO keyword package."""

        self._log(run_id, "Generating SEO package with Groq LLM...", "INFO")
        response_text = self.call_llm_with_retry(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
            run_id=run_id,
        )

        try:
            keyword_data = self.parse_llm_json(response_text)
        except json.JSONDecodeError:
            # Fallback with basic keyword structure
            keyword_data = {
                "primary_keyword": topic,
                "secondary_keywords": all_suggestions[:5] if all_suggestions else [topic],
                "tags": (all_suggestions[:15] if len(all_suggestions) >= 15
                         else all_suggestions + [niche] * (15 - len(all_suggestions))),
                "hashtags": [f"#{niche}", f"#{topic.replace(' ', '')}", "#youtube", "#trending", "#viral"],
                "search_intent": f"Learn about {topic}",
                "keyword_clusters": {"informational": [topic]},
            }
            self._log(run_id, "Using fallback keyword structure due to JSON parse error", "WARNING")

        # Ensure lists have correct lengths
        keyword_data.setdefault("tags", [topic, niche])
        keyword_data.setdefault("hashtags", [f"#{niche}"])
        if len(keyword_data.get("tags", [])) > 15:
            keyword_data["tags"] = keyword_data["tags"][:15]

        self._log(run_id, f"Primary keyword: '{keyword_data.get('primary_keyword')}', {len(keyword_data.get('tags', []))} tags generated", "INFO")
        return {"keyword_data": keyword_data}
