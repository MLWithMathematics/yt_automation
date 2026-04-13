# YouTube Agent Pipeline

A fully automated multi-agent YouTube content creation system.  
From trend discovery → script → voiceover → B-roll → video → thumbnail → metadata.  
All free tools. Real-time dashboard. Resume-on-failure support.

---

## Quick Start

### 1. Clone & Install

```bash
git clone <repo>
cd youtube_agent
python3 -m venv venv && source venv\bin\activate
pip install -r requirements.txt
```

Install FFmpeg (required for video assembly):
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html and add to PATH
```

### 2. Get Free API Keys

**Groq API (Required — for all LLM calls)**
1. Go to https://console.groq.com
2. Sign up for a free account
3. Create an API key under "API Keys"
4. Free tier: 14,400 req/day with llama-3.3-70b-versatile

**YouTube Data API v3 (Optional — improves trend detection)**
1. Go to https://console.cloud.google.com
2. Create a new project → Enable "YouTube Data API v3"
3. Create credentials → API Key
4. Free quota: 10,000 units/day (enough for several runs/day)

**Pexels API (Optional — for B-roll images)**
1. Go to https://www.pexels.com/api/
2. Sign up for a free account
3. Create an API key
4. Free tier: 200 req/hour, 20,000 req/month
5. If not configured, falls back to Pollinations.ai (fully free, no key)

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your API keys:
GROQ_API_KEY=your_groq_key_here
YOUTUBE_API_KEY=your_youtube_key_here   # optional
PEXELS_API_KEY=your_pexels_key_here     # optional
```

Edit `config.py` to set your niche and preferences:
```python
NICHE = "technology"          # Your content niche
VIDEO_LENGTH = "medium"       # short (3min) | medium (7min) | long (12min)
VOICE = "en-US-GuyNeural"     # edge-tts voice
SCHEDULER_INTERVAL_HOURS = 24 # Auto-run interval
```

### 4. (Optional) Add Assets

```bash
# Add your channel logo (for thumbnail watermark)
cp your_logo.png assets/logo.png

# Add royalty-free background music
cp your_music.mp3 assets/background.mp3

# Add Impact.ttf for thumbnail text
cp Impact.ttf assets/Impact.ttf
```

### 5. Run

```bash
python main.py
```

Open your browser at **http://localhost:8000**

---

## Dashboard Features

| Panel | Description |
|-------|-------------|
| **Top Bar** | Start runs, view config |
| **Left Panel** | Pipeline flowchart with live agent status |
| **Center Panel** | Real-time scrolling log feed with SSE |
| **Right Panel** | Thumbnail preview, metadata, video player, downloads |
| **Bottom Bar** | Full run history with pagination |

---

## Resuming a Failed Run

### Via Dashboard
1. Find the failed run in the Run History table
2. Click **[Resume]** — pipeline restarts from the failed agent

### Via API
```bash
# Resume a run from where it failed
curl -X POST http://localhost:8000/api/runs/{run_id}/resume

# Retry a specific agent only
curl -X POST http://localhost:8000/api/runs/{run_id}/agents/VideoAgent/retry
```

### How Resume Works
- Every agent saves its complete output to SQLite on completion
- On resume: all completed agents' outputs are loaded from DB
- LangGraph pipeline is rebuilt starting from the first incomplete agent
- Example: If VideoAgent failed, resume skips Trend→Keyword→Script→Voiceover→Visual
  and resumes directly at VideoAgent with all prior data intact

---

## Agent Pipeline

| # | Agent | What It Does |
|---|-------|-------------|
| 1 | **TrendAgent** | Google Trends + YouTube trending → AI picks best topic |
| 2 | **KeywordAgent** | YouTube autocomplete scraping → SEO keyword package |
| 3 | **ScriptAgent** | Groq LLM generates full retention-optimized script |
| 4 | **VoiceoverAgent** | edge-tts converts script → MP3 with timestamp map |
| 5 | **VisualAgent** | Pexels/Pollinations.ai fetches B-roll per section |
| 6 | **VideoAgent** | Whisper subtitles + MoviePy assembly + FFmpeg burn |
| 7 | **ThumbnailAgent** | Pollinations.ai + Pillow composites final thumbnail |
| 8 | **MetadataAgent** | Groq LLM generates title, description, tags, hashtags |

---

## Output Files

Each run creates `output/run_{id}/`:
```
output/run_{id}/
├── voiceover.mp3       # TTS audio
├── visuals/            # Downloaded B-roll images
├── subtitles.srt       # Whisper-generated subtitles
├── video_raw.mp4       # Intermediate video
├── final_video.mp4     # Final video with subtitles
└── thumbnail.jpg       # YouTube thumbnail (1280x720)
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/api/runs/start` | POST | Start new run (`?niche=technology`) |
| `/api/runs` | GET | List all runs (`?limit=10&offset=0`) |
| `/api/runs/{id}` | GET | Run details + tasks + logs |
| `/api/runs/{id}/resume` | POST | Resume failed run |
| `/api/runs/{id}/agents/{name}/retry` | POST | Retry single agent |
| `/api/runs/{id}/output` | GET | Output file paths |
| `/api/runs/{id}/logs` | GET | Agent logs (`?agent_name=VideoAgent`) |
| `/api/sse/{run_id}` | GET | SSE stream for run events |
| `/api/health` | GET | Health check |

---

## Troubleshooting

**Groq rate limits (429)**
- Automatically waits 60s and retries — not counted as failure

**edge-tts network errors**
- Retried 3× with 10s gap before marking failed

**FFmpeg not found**
- Install FFmpeg and ensure it's in your PATH

**Whisper model download**
- First run downloads the model (~150MB for "base")
- Subsequent runs use cached model

**Pexels quota exceeded**
- System automatically falls back to Pollinations.ai (free, no key)

**MoviePy errors**
- Check `pipeline.log` for full FFmpeg stderr output

---

## Environment Variables

```
GROQ_API_KEY      Required. Groq API key for LLM calls.
YOUTUBE_API_KEY   Optional. YouTube Data API v3 key.
PEXELS_API_KEY    Optional. Pexels API key for B-roll.
```

---

## Tech Stack (All Free)

- **LangGraph** — Agent orchestration graph
- **Groq** — LLM API (llama-3.3-70b-versatile, free tier)
- **edge-tts** — Microsoft neural TTS (free)
- **pytrends** — Google Trends scraping
- **YouTube Data API v3** — Trending video data
- **Pexels API** — Stock photos
- **Pollinations.ai** — AI image generation (free, no key)
- **openai-whisper** — Local speech-to-text for subtitles
- **MoviePy + FFmpeg** — Video assembly
- **Pillow** — Thumbnail compositing
- **FastAPI + SSE** — Real-time dashboard backend
- **SQLite + SQLAlchemy** — Persistent state
- **APScheduler** — Automated scheduling
