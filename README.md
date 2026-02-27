# MathVizAI MVP

An LLM-driven system that converts mathematical topics and research paper excerpts into high-quality animated visual explanations (3Blue1Brown style).

## Features
- Input a math topic (e.g., "Central Limit Theorem") or upload a short PDF (≤10 pages)
- Automatically generates a pedagogically structured 3–5 scene animation plan
- Produces and renders Manim animation code
- Outputs a 2–4 minute educational video

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key
- Manim Community Edition

### Installation
```bash
# 1. Clone / unzip the project
cd mathvizai

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Manim
pip install manim

# 4. Set environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Start the backend
uvicorn backend.api.main:app --reload --port 8000

# 6. Start the Streamlit frontend (new terminal)
streamlit run frontend/app.py
```

### Docker (recommended)
```bash
docker-compose up --build
```

## Architecture

```
User Input
    ↓
Paper/Topic Parser Agent
    ↓
Concept Extraction Agent
    ↓
Pedagogical Planner Agent
    ↓
Scene Generator Agent
    ↓
Animation Code Generator Agent
    ↓
Renderer Module
    ↓
Final Video Output
```

## API Endpoints
- `POST /generate-video` — full pipeline
- `GET /status/{job_id}` — poll job status
- `GET /download/{job_id}` — download rendered video

## Configuration
All settings live in `backend/config.py`. Key options:
- `OPENAI_MODEL` — default `gpt-4o`
- `MAX_SCENES` — default `5`
- `RENDER_QUALITY` — default `medium_quality`

## Running Tests
```bash
pytest tests/ -v
```
