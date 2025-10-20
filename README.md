# Rick & Morty AI Challenge

This project is a lightweight Streamlit app demonstrating retrieval, reasoning, generation, and evaluation over the Rick & Morty API.

## Why GraphQL over REST
- GraphQL lets us fetch nested structures (locations with residents and key character fields) in a single round-trip and shape the response precisely.
- REST requires multiple paginated endpoints and N+1 follow-ups to get resident details. GraphQL reduces over-fetching and under-fetching while improving developer ergonomics via a self-documented schema.
- Trade-offs: GraphQL queries can be larger and require some pagination awareness; error surfaces can be more complex. For this dataset size, the benefits outweigh costs.

## Persistence choice: SQLite
- SQLite offers a zero-config, file-based database with strong consistency and SQL capabilities—perfect for local persistence of character notes and optional embeddings.
- Alternatives like JSON are simpler but become unwieldy for querying, ranking, and semantic search. SQLite plus a small DAO gives us reliability and room to grow.

## Features
- Browse locations with their type and residents (name, status, species, image).
- View character details; add persistent notes per character.
- Generative feature: produce a location summary in a Rick & Morty narrator tone or generate a short dialogue between two characters.
- Lightweight evaluation: heuristic rubric (consistency, creativity, completeness) and optional embedding similarity.
- Bonus: AI-augmented search across characters and notes via TF‑IDF.

## Setup
1. Python >= 3.9 recommended.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. (Optional) Set environment variables for LLMs in a `.env` file:
```bash
OPENAI_API_KEY=sk-...
```

## Run
```bash
streamlit run app.py
```

## Files
- `app.py`: Streamlit UI.
- `data_client.py`: GraphQL data retrieval with light caching.
- `db.py`: SQLite persistence for notes (and optional embeddings).
- `gen.py`: Generative functions with provider fallback.
- `eval.py`: Heuristic and embedding-based evaluation scaffolding.

## Demo artifacts
Please record a short GIF and a brief video walkthrough of the app:
- GIF: key flows (browse locations, open a character, add a note, generate + evaluate summary, search).
- Video: narrate design choices and demonstrate usage.

## Notes on evaluation design
- Factual consistency: checks presence of referenced entity names and fields against known facts.
- Creativity: uses type-token ratio and lexical diversity as a rough heuristic.
- Completeness: checks coverage of required fields for the prompt (e.g., mentions of location type, count of residents, notable species).
- Optional embedding similarity: cosine similarity via TF‑IDF between generation and a fact sheet built from the structured data.

## Limitations
- Heuristics are coarse and meant for demonstration. For production, consider task-specific rubrics, human-in-the-loop ratings, and more robust grounding.
