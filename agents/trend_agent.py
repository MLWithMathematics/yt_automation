"""
agents/trend_agent.py — Agent 1: Discovers trending video topics.
"""
import json
import time
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
import config
import database.db as db

logger = logging.getLogger(__name__)


class TrendAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="TrendAgent",
            role_description=(
                "Discovers trending video topics in the configured niche by combining "
                "Google Trends data and YouTube trending videos, then uses AI to rank "
                "the top ideas by virality, feasibility, and uniqueness."
            ),
            inputs_expected=["niche"],
            outputs_produced=["trend_data"],
        )

    def _get_google_trends(self, niche: str, run_id: str) -> list:
        """Pull rising queries from pytrends."""
        try:
            from pytrends.request import TrendReq
            pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
            pt.build_payload([niche], cat=0, timeframe=config.PYTRENDS_TIMEFRAME, geo="US")
            related = pt.related_queries()
            rising = related.get(niche, {}).get("rising")
            if rising is not None and not rising.empty:
                return rising["query"].tolist()[:10]
        except Exception as e:
            self._log(run_id, f"pytrends error: {e}", "WARNING")
        return []

    def _get_youtube_trending(self, niche: str, run_id: str) -> list:
        """Pull trending YouTube videos via YouTube Data API v3."""
        if not config.YOUTUBE_API_KEY:
            self._log(run_id, "No YOUTUBE_API_KEY configured, skipping YouTube trending", "WARNING")
            return []
        try:
            from googleapiclient.discovery import build
            youtube = build("youtube", "v3", developerKey=config.YOUTUBE_API_KEY)
            category_id = str(config.get_category_id())
            response = youtube.videos().list(
                part="snippet,statistics",
                chart="mostPopular",
                regionCode=config.REGION_CODE,
                videoCategoryId=category_id,
                maxResults=20,
            ).execute()
            titles = []
            for item in response.get("items", []):
                title = item["snippet"]["title"]
                titles.append(title)
            return titles
        except Exception as e:
            self._log(run_id, f"YouTube API error: {e}", "WARNING")
            return []

    def _get_recent_topics(self, run_id: str) -> list:
        """Get topics from last 10 pipeline runs to avoid repeats."""
        try:
            runs = db.list_runs(limit=10)
            topics = []
            for run in runs:
                output = db.get_task_output(run["id"], "TrendAgent")
                if output and "topic" in output:
                    topics.append(output["topic"])
            return topics
        except Exception:
            return []

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        run_id = state["run_id"]
        niche = state["niche"]

        self._log(run_id, f"Fetching Google Trends for niche: {niche}", "INFO")
        trending_queries = self._get_google_trends(niche, run_id)
        self._log(run_id, f"Google Trends returned {len(trending_queries)} rising queries", "INFO")

        self._log(run_id, "Fetching YouTube trending videos...", "INFO")
        youtube_titles = self._get_youtube_trending(niche, run_id)
        self._log(run_id, f"YouTube API returned {len(youtube_titles)} trending videos", "INFO")

        recent_topics = self._get_recent_topics(run_id)
        self._log(run_id, f"Recent topics to avoid: {recent_topics}", "INFO")

        # Use Groq to rank and select best topic
        system_prompt = f"""You are a YouTube content strategist specialized in {niche}.
Your task: Analyze trending data and select the BEST video topic.
Scoring criteria:
1. Virality potential (search volume + emotional appeal)
2. Content feasibility (can be covered in {config.VIDEO_LENGTH} format)
3. Uniqueness (NOT similar to: {recent_topics})
4. Current relevance (trending NOW)

Return ONLY valid JSON, no markdown, no explanation:
{{
  "topic": "specific video topic",
  "angle": "unique perspective/angle to cover it",
  "target_audience": "who this video is for",
  "trend_score": 0.0-1.0,
  "reasoning": "why this topic was chosen"
}}"""

        user_prompt = f"""Niche: {niche}
Google Trends rising queries: {json.dumps(trending_queries)}
YouTube trending video titles: {json.dumps(youtube_titles[:20])}
Recent topics already covered (avoid these): {json.dumps(recent_topics)}

Select the single best video topic from this data."""

        self._log(run_id, "Ranking topics with Groq LLM...", "INFO")
        response_text = self.call_llm_with_retry(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
            run_id=run_id,
        )

        try:
            trend_data = self.parse_llm_json(response_text)
        except json.JSONDecodeError:
            # Fallback: pick from trending_queries or youtube_titles
            fallback_topic = (trending_queries[0] if trending_queries
                              else (youtube_titles[0] if youtube_titles else f"{niche} trends 2025"))
            trend_data = {
                "topic": fallback_topic,
                "angle": f"Everything you need to know about {fallback_topic}",
                "target_audience": f"People interested in {niche}",
                "trend_score": 0.6,
                "reasoning": "LLM parse failed, using top trending item as fallback",
            }
            self._log(run_id, f"JSON parse failed, using fallback topic: {fallback_topic}", "WARNING")

        self._log(run_id, f"Selected topic: {trend_data.get('topic')} (score: {trend_data.get('trend_score')})", "INFO")
        return {"trend_data": trend_data}
