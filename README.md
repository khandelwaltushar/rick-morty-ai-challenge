# Rick & Morty AI Challenge

A lightweight Streamlit app demonstrating retrieval, reasoning, generation, evaluation, and AI-augmented search over the Rick & Morty API.

## Overview
- Browse locations with their type, dimension, and residents (name, status, species, image)
- View character details and add persistent notes (SQLite)
- LLM-powered generation: narrator-style location summaries and two-character dialogue
- Lightweight evaluation: factual consistency, creativity, completeness, and grounding similarity
- AI-augmented search: semantic retrieval across notes and residents via embeddings (OpenAI) with a robust local fallback

## Why GraphQL over REST
- **Single query for nested data**: We need locations and their residents including name, status, species, and image. With GraphQL we fetch this in one query across pages.
- **Developer ergonomics**: GraphQL schema self-documents fields and relationships, reducing over/under-fetching.
- **Trade-offs**: Slightly larger queries and pagination handling; error surfaces sometimes more verbose. For this use case, the reduced N+1 REST calls and exact field selection make GraphQL preferable.

## Architecture
- `app.py`: Streamlit UI. Tabs for summary generation, dialogue, and semantic search.
- `data_client.py`: GraphQL client with pagination and a tiny JSON on-disk cache for locations.
- `db.py`: SQLite persistence for notes with a minimal DAO layer.
- `gen.py`: Generative functions using OpenAI Chat Completions with a safe, witty narrator system prompt; includes template fallbacks.
- `eval.py`: Heuristic evaluation of generations.
- `embeddings.py`: Embedding utilities. Uses OpenAI embeddings when available; otherwise a robust character n‑gram TF‑IDF + SVD fallback.

### Data retrieval details
- Endpoint: `https://rickandmortyapi.com/graphql`
- Query: paginated `locations(page: $page) { results { id, name, type, dimension, residents { id, name, status, species, image, gender, origin { name } } } }`
- Caching: JSON file `.cache_locations.json` to avoid repeated network calls during local dev.

### Persistence details (SQLite)
- File: `notes.db`
- Schema: `notes(id INTEGER PK, character_id TEXT, character_name TEXT, content TEXT, embedding BLOB, created_at TEXT)`
- Index: `idx_notes_character` on `character_id`
- Motivations: durability, simple querying, and room to add embeddings storage later.

### Generative layer
- Client: OpenAI (via `openai` SDK) if `OPENAI_API_KEY` is set.
- Models: `gpt-4o-mini` for chat; temperature tuned for creative output.
- Prompts:
  - Location summary: 4–6 sentences, narrator tone, factual grounding (type, dimension, notable residents)
  - Dialogue: 6–8 lines between two chosen characters, on-brand but safe
- Fallbacks: If no API key or an error occurs, deterministic templates produce readable output.

### Evaluation scaffolding
- **Factual consistency**: counts overlap with tokens derived from known facts (location name, type, dimension, resident names)
- **Creativity**: lexical diversity (type-token ratio) mapped into [0,1]
- **Completeness**: coverage of required keywords (location name, type, dimension)
- **Grounding similarity**: cosine similarity between generated text and a fact sheet using TF‑IDF
- **Overall**: mean of the above; intended as a simple, transparent proxy—not a gold standard.

### AI‑augmented search (Semantic / Fuzzy)
- Primary mode: **OpenAI embeddings** (`text-embedding-3-small`) with cosine similarity, if `OPENAI_API_KEY` is present.
- Fallback mode: **Character n‑gram TF‑IDF (3–5) + TruncatedSVD** to produce dense, normalized vectors; cosine similarity for ranking.
- Why character n‑grams: robust to misspellings and variants (e.g., "gutterman" ≈ "Guetermann").
- Indexed corpus: all notes + synthetic resident entries (name, species, status, and current location context) for better coverage.

## Setup
1. Python 3.9+ recommended.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. (Optional) Create `.env` for LLMs:
```bash
OPENAI_API_KEY=sk-...
```

## Run
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`.

## Usage Guide
- **Locations**: Use the sidebar to select a location; residents are shown with images and basic info.
- **Notes**: Add notes to any character; persisted in `notes.db`.
- **Summarize**: Click "Generate summary"; scores appear beneath the output.
- **Dialogue**: Select two characters and generate a short exchange.
- **Search**: Enter a free-text query. Results are ranked by cosine similarity in the embedding space; works for fuzzy queries.

## Design choices and trade-offs
- **Streamlit**: Rapid UI development and easy local demos.
- **GraphQL**: Reduced N+1 calls and exact field selection at the cost of handling pagination; worth it here.
- **SQLite vs JSON**: Chose SQLite for querying, indexing, and future-proofing (embedding storage) over simpler JSON files.
- **LLM fallbacks**: Ensures the app remains functional offline or without API keys.
- **Evaluation**: Heuristic by design—transparent and adjustable. In production, combine with human ratings or task-specific rubrics.
- **Embeddings fallback**: Character n‑grams with SVD trade some semantic richness for robustness, determinism, and zero external calls.

## Demo artifacts (what to include in submission)
- **GIF**: Show browsing locations, adding a note, generating + evaluating a summary, and semantic search.
- **Video**: Short walkthrough of architecture and trade-offs; show both LLM-enabled and fallback behavior.

## Repository hygiene
- `.gitignore` excludes `.venv`, `notes.db`, `.env`, caches, and Streamlit state.

## Troubleshooting
- Slow first load: GraphQL paging + caching. Subsequent loads use `.cache_locations.json`.
- All search scores near zero: Either the corpus is tiny or the previous fallback used word tokens. The current fallback uses char n‑grams to avoid this. If you have an API key set, OpenAI embeddings will typically improve results further.
- macOS file watcher: Streamlit suggests `xcode-select --install` and `pip install watchdog` for better hot-reload performance.
