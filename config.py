"""
config.py — Central configuration for YouTube Agent Pipeline.
"""
import os
from dotenv import load_dotenv
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str          = os.getenv("GROQ_API_KEY", "")
YOUTUBE_API_KEY: str       = os.getenv("YOUTUBE_API_KEY", "")
PEXELS_API_KEY: str        = os.getenv("PEXELS_API_KEY", "")
HUGGINGFACE_API_KEY: str   = os.getenv("HUGGINGFACE_API_KEY", "")

# ── Content Settings ──────────────────────────────────────────────────────────
NICHE: str                 = "technology"
VIDEO_LENGTH: str          = "short"        # short | medium | long
LANGUAGE: str              = "en"
REGION_CODE: str           = "US"
YOUTUBE_CATEGORY_ID: int   = 28

VOICE: str                 = "en-US-GuyNeural"

VIDEO_LENGTH_SECONDS: dict = {
    "short":  120,   # 2 min
    "medium": 420,   # 7 min
    "long":   720,   # 12 min
}

# ── LLM + Whisper ─────────────────────────────────────────────────────────────
GROQ_MODEL: str            = "llama-3.1-8b-instant"  # llama-3.1-8b-instant & llama-3.3-70b-versatile
WHISPER_MODEL: str         = "base"          # tiny | base | small

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR: str              = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR: str            = os.path.join(BASE_DIR, "output")
ASSETS_DIR: str            = os.path.join(BASE_DIR, "assets")
BG_MUSIC_PATH: str         = os.path.join(ASSETS_DIR, "background.mp3")
CHANNEL_LOGO_PATH: str     = os.path.join(ASSETS_DIR, "logo.png")
THUMBNAIL_FONT_PATH: str   = os.path.join(ASSETS_DIR, "Impact.ttf")
FALLBACK_FONT_PATH: str    = os.path.join(ASSETS_DIR, "Oswald-Bold.ttf")
DATABASE_URL: str          = f"sqlite:///{os.path.join(BASE_DIR, 'pipeline.db')}"

# ── Pipeline Control ──────────────────────────────────────────────────────────
MAX_AGENT_RETRIES: int     = 3
RETRY_BACKOFF_SECONDS: list= [5, 15, 45]
SCHEDULER_INTERVAL_HOURS: int = 24

# ── Video Dimensions ──────────────────────────────────────────────────────────
VIDEO_WIDTH: int           = 1920
VIDEO_HEIGHT: int          = 1080
THUMBNAIL_WIDTH: int       = 1280
THUMBNAIL_HEIGHT: int      = 720

# ── Visual / B-Roll ───────────────────────────────────────────────────────────
PEXELS_PER_SECTION: int    = 5
BG_MUSIC_VOLUME: float     = 0.12

# ── Video Assembly ────────────────────────────────────────────────────────────
CROSSFADE_DURATION: float  = 0.5   # seconds; set 0.0 to disable
KEN_BURNS_ZOOM: float      = 0.06  # 0.0 = off, 0.06 = subtle, 0.15 = strong

# ── AnimateDiff ───────────────────────────────────────────────────────────────
ENABLE_ANIMATEDIFF: bool   = False
ANIMATEDIFF_ADAPTER: str   = "guoyww/animatediff-motion-adapter-v1-5-2"
ANIMATEDIFF_BASE_MODEL: str= "runwayml/stable-diffusion-v1-5"
ANIMATEDIFF_NUM_FRAMES: int= 16
ANIMATEDIFF_STEPS: int     = 20
ANIMATEDIFF_GUIDANCE: float= 7.5
ANIMATEDIFF_FPS_OUT: int   = 8
ANIMATEDIFF_WIDTH: int     = 512
ANIMATEDIFF_HEIGHT: int    = 512
ANIMATEDIFF_UNLOAD_AFTER_CLIP: bool = True

# ── Pytrends ──────────────────────────────────────────────────────────────────
PYTRENDS_TIMEFRAME: str    = "now 7-d"
TREND_TOP_N: int           = 5

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_HOST: str        = "0.0.0.0"
DASHBOARD_PORT: int        = 8000

# ── Metadata helpers ──────────────────────────────────────────────────────────
POWER_WORDS: list          = [
    "Shocking", "Secret", "Why", "How", "Inside",
    "Never", "Finally", "Exposed", "Revealed", "Truth"
]

NICHE_CATEGORY_MAP: dict   = {
    "technology": 28, "science": 28, "finance": 25,
    "gaming": 20,     "health": 26,  "education": 27,
    "entertainment": 24, "news": 25, "people": 22,
    "travel": 19,     "food": 26,    "sports": 17,
    "kids": 20,       "cartoon": 1,  "animation": 1,
}


def get_target_duration() -> int:
    return VIDEO_LENGTH_SECONDS.get(VIDEO_LENGTH, 180)


def get_category_id() -> int:
    return NICHE_CATEGORY_MAP.get(NICHE.lower(), YOUTUBE_CATEGORY_ID)
ENABLE_SVD_ANIMATION = False