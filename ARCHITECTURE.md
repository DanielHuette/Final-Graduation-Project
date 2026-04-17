# System Architecture — Multilingual YouTube Analyzer

**Author:** DanielHuette  
**Version:** 1.0

---

## Overview

The Multilingual YouTube Analyzer is composed of two Google Colab notebooks that share a common data layer (`benchmark_results/`). The benchmark notebook produces configuration exports that are consumed by the analyzer at runtime, creating a clean separation between the experimental phase and the production application.

```
benchmark.ipynb  ──exports──►  benchmark_results/  ──loads──►  youtube_analyzer.ipynb
```

---

## Ingestion Pipeline

```
YouTube URL
    │
    ▼ yt-dlp
Audio file (.m4a, 64 kbps)
    │
    ▼ Whisper (API or local)
Raw transcript text + segment timestamps
    │
    ▼ tiktoken RecursiveCharacterTextSplitter
    │   chunk_size + chunk_overlap from chunk_recommendations.json
    │   (adaptive: selected per video duration × Whisper model)
Chunks with metadata (video_id, title, channel, url,
                       chunk_index, start_sec, end_sec,
                       start_formatted, end_formatted,
                       has_timestamps, language)
    │
    ▼ text-embedding-3-small (OpenAI)
Dense vectors (1536 dimensions)
    │
    ▼ ChromaDB (local persist or Google Drive)
Vector store collection: "youtube_videos"
```

### Adaptive Chunking

At ingestion time, `get_chunk_config(duration_sec, whisper_option)` resolves the optimal chunk size and overlap from `chunk_recommendations.json` based on:

1. **Video bucket:** `short_video` (≤600 s), `medium_video` (600–2400 s), `long_video` (>2400 s)
2. **Whisper model:** `api`, `tiny`, `base`, `small`, `medium`, `large-v3`

If the JSON file is not present, a `RuntimeError` is raised — no silent fallback, ensuring configuration integrity.

---

## Agent Architecture

The agent is implemented as a LangGraph **ReAct agent** (`create_react_agent`) with persistent per-session memory via `MemorySaver`.

```
User message
    │
    ▼ build_system_prompt(active_video_id, active_video_title)
SYSTEM_PROMPT with active video context
    │
    ▼ ChatOpenAI (configurable model)
    │
    ├──► search_video_content(query)      — ChromaDB similarity search, filtered to active video
    ├──► list_indexed_videos()            — List all ChromaDB entries
    ├──► add_new_video(url)               — Full ingestion pipeline
    ├──► get_video_metadata_tool(id)      — Metadata lookup
    ├──► summarize_video(id)              — Full-transcript summarisation
    ├──► compare_videos(id1,id2)          — Side-by-side transcript comparison
    └──► extract_key_moments(id)          — Key moment extraction
    │
    ▼ MemorySaver (thread_id per session)
Conversation history persisted across turns
    │
    ▼ Agent response
```

### Active Video State

The UI maintains an `active_video_state` (Gradio `gr.State`) that tracks the most recently interacted video. This state is injected into:

1. The **SYSTEM_PROMPT** via `build_system_prompt()` — instructs the agent to restrict tool calls to the active video
2. **`search_video_content`** — applies a ChromaDB `filter={"video_id": active_video_id}` directly, without relying on agent compliance
3. **`ask_agent_with_usage`** — triggers agent rebuild when the active video changes

The agent is rebuilt (new `create_react_agent` instance) whenever the active video or model changes, ensuring the system prompt reflects the current context.

---

## Gradio UI Architecture

The UI is structured around five tabs:

| Tab | Content |
|-----|---------|
| 💬 Chat Interface Hub | Two video inputs, chat panel, cost panel, compare output, top passages |
| ✅ Judgement Day | WER check (Whisper vs. YouTube captions), LLM-as-Judge faithfulness eval |
| 🏆 Benchmark | Sweet-spot cards, sortable Whisper × LLM combo table |
| 📊 Benchmark Plots | Visualisation plots from `benchmark_results/` |
| ℹ️ Credits & Github | Credits |

### Active Video Rule

The **last interaction wins** principle governs which video the agent answers about:
- Selecting a video in either dropdown → sets `active_video_state`
- Indexing a new video → automatically selects it in the corresponding dropdown → sets `active_video_state`
- Initial load → `active_video_state` is initialised to the last indexed video in ChromaDB

### Cost Tracking

All costs are computed locally from token counts returned by the OpenAI callback (`get_openai_callback`) and multiplied by prices from `prices.json`. LangSmith provides independent tracing but is not the source for displayed costs.

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    benchmark_results/                        │
│  prices.json  combo_table.csv  chunk_recommendations.json   │
│  winners.json  summary.json  *.png                          │
└──────────────────────┬──────────────────────────────────────┘
                       │ loaded at startup (Cell 12)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    BENCHMARK_DATA dict                       │
│  llm_prices  whisper_prices  embedding_prices               │
│  combo_table  winners  summary  chunk_recommendations        │
└──────┬──────────────────────────────────────────────────────┘
       │
       ├── get_chunk_config()        → adaptive chunk size at ingest
       ├── get_llm_price()           → cost panel calculations
       ├── get_whisper_price()       → cost panel calculations
       ├── _get_combo_df()           → benchmark tab table
       ├── _SWEET_SPOTS              → benchmark tab cards
       └── _find_plot()              → benchmark plots tab images
```

---

## Security Considerations

- API keys are never hardcoded. They are loaded exclusively from Colab Secrets, environment variables, or a local `.env` file (not committed to version control).
- The `.env.example` file contains only placeholder values.
- ChromaDB data is local to the Colab session by default. Google Drive persistence is opt-in via `CONFIG["use_drive"]`.

---

## Technology Stack

| Component | Library / Service | Version |
|-----------|-------------------|---------|
| Audio download | `yt-dlp` | latest |
| Transcription | `openai-whisper`, OpenAI API | `whisper-1` |
| Text splitting | `langchain-text-splitters` | ≥0.3 |
| Tokenisation | `tiktoken` | latest |
| Vector store | `chromadb`, `langchain-chroma` | latest |
| Embeddings | `langchain-openai` (`text-embedding-3-small`) | latest |
| Agent framework | `langgraph` (`create_react_agent`) | latest |
| Memory | `langgraph` (`MemorySaver`) | latest |
| LLM (OpenAI) | `langchain-openai` (`ChatOpenAI`) | latest |
| LLM (HuggingFace) | `langchain-openai` (HF Inference API) | latest |
| Tracing | `langsmith` (EU endpoint) | latest |
| UI | `gradio` | latest |
| WER | `jiwer` | latest |
| YouTube captions | `youtube-transcript-api` | latest |
