# Atlas - AI Chat Assistant

A conversational AI assistant built with LangGraph and Gradio that provides three specialized services: weather information, music discovery, and calculations.

## Features

### Three Services

1. **Weather Service (API Calls)**
   - Uses [wttr.in](https://wttr.in) - a free weather API requiring no API key
   - Transforms JSON response into natural language descriptions
   - Example: "What's the weather in Toronto?"

2. **Music Discovery (Semantic Search)**
   - Semantic search over Pitchfork album reviews
   - Uses ChromaDB with persistent file storage
   - OpenAI `text-embedding-3-small` for embeddings
   - Example: "Find me albums similar to Radiohead"

3. **Calculator (Function Calling)**
   - Mathematical expression evaluation
   - Unit conversions (temperature, distance, weight, volume)
   - Example: "Calculate sqrt(144) + 10" or "Convert 100 km to miles"

### Personality

Atlas has a friendly, slightly witty personality with a dry sense of humor. Responses are concise and helpful.

### Guardrails

The system includes safety measures:
- **Topic Restrictions**: Will not discuss cats, dogs, horoscopes, zodiac signs, or Taylor Swift
- **Prompt Protection**: Refuses to reveal or discuss system instructions
- **Injection Detection**: Detects and refuses prompt injection attempts

### Memory Management

The system implements a summarization-based memory strategy to handle long conversations:

**Configuration:**
- `MAX_HISTORY_MESSAGES = 20` — Threshold to trigger summarization
- `KEEP_RECENT_MESSAGES = 10` — Recent messages preserved after summarization

**How it works:**
1. Conversation history is maintained throughout the session
2. When messages exceed 20, older messages are summarized using the LLM
3. Summary is prepended as context: `[Previous conversation summary: ...]`
4. 10 most recent messages remain intact for immediate context
5. Falls back to simple truncation if summarization fails

**Why this approach:**
- Preserves important context from early conversation
- Reduces token usage while maintaining coherence
- Follows LangGraph's recommended pattern for short-term memory management

**Example flow:**
```
Messages 1-15: Normal conversation
Message 16-20: Still within limit
Message 21+: Messages 1-10 summarized → [Summary] + Messages 11-21
```

## Setup

### Run the Application

The ChromaDB vector database is pre-built and included in the repository (`chroma_data/` folder).

```bash
cd 05_src
python -m assignment_chat.app
```

The Gradio interface will launch at `http://127.0.0.1:7860`.

### (Optional) Rebuild Embeddings

If you need to regenerate the embeddings:

```bash
cd 05_src
python -m assignment_chat.create_embeddings
```

This loads Pitchfork reviews from `documents/pitchfork_content.jsonl` and stores embeddings in `assignment_chat/chroma_data/`.

## File Structure

```
assignment_chat/
├── __init__.py           # Package marker
├── app.py                # Gradio chat interface
├── main.py               # LangGraph agent logic
├── prompts.py            # System prompt and guardrails
├── tools.py              # Tool implementations (weather, music, calc)
├── create_embeddings.py  # Script to rebuild ChromaDB (optional)
├── chroma_data/          # Pre-built ChromaDB storage (included in repo)
└── readme.md             # This file
```

## Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| **wttr.in API** | Free, no API key required, returns comprehensive weather data |
| **ChromaDB Persistent Client** | File-based storage as required, no Docker dependency |
| **LangGraph StateGraph** | Follows course patterns, handles tool calling loop |
| **Guardrails in Python** | Pre-filters messages before LLM call for efficiency |
| **Summarization-based memory** | Preserves context while managing token limits (20 msg threshold, keeps 10 recent + summary) |

## Embedding Process

**Note:** Embeddings are pre-built and included in `chroma_data/`. No action required.

The `create_embeddings.py` script (for reference):
1. Loads reviews from `pitchfork_content.jsonl` (JSONL format)
2. Uses OpenAI `text-embedding-3-small` via course API gateway
3. Stores embeddings in ChromaDB with metadata (artist, title, score)
4. Persists to `chroma_data/` directory

## Dependencies

Uses standard course environment:
- `langchain`, `langgraph`
- `gradio`
- `chromadb`
- `openai`
- `requests`
- `python-dotenv`

## Example Interactions

```
User: What's the weather like in London?
Atlas: Current weather in London, United Kingdom: Partly cloudy with a temperature of 15°C (feels like 14°C). Humidity is at 72% with winds from the SW at 12 km/h.

User: Find me some 10/10 rated albums
Atlas: **Radiohead - OK Computer** (Score: 10/10)
Radiohead's OK Computer is a landmark album that captures the anxiety of modern life...

User: What is 25 * 4 + sqrt(100)?
Atlas: 110

User: Tell me about cats
Atlas: I'm not able to discuss that topic. How can I help you with weather, music, or calculations instead?
```
