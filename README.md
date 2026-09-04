# Support Call Pipeline (Demo)

A small end-to-end voice pipeline for a customer-support scenario:

```
audio in → preprocess (mono/16kHz) → ASR transcript
        → embed + retrieve similar past calls
        → LLM answer (summary, intent, next action)
        → TTS spoken answer
```

Built as a teaching demo — eight clearly separated steps, one function
(or small pair of functions) per step, orchestrated by `main()`.

## Pipeline steps → functions

| # | Step | Function(s) |
|---|---|---|
| 1 | Setup and verification | `verify_environment()` |
| 2 | Prepare a short audio file | *(external — place a recording at `INPUT_AUDIO`)* |
| 3 | Load and preprocess audio (mono, 16kHz) | `preprocess_audio()` |
| 4 | Transcribe the audio (ASR) | `transcribe_audio()` via `get_transcript()` |
| 5 | Build a small retrieval dataset | `PAST_CALLS` |
| 6 | Embed and retrieve similar calls | `embed_texts()`, `normalize_rows()`, `retrieve_similar_calls()` |
| 7 | Generate an LLM answer | `generate_answer()` |
| 8 | Convert the answer back into speech | `clean_for_tts()`, `synthesize_speech()` |

## Requirements

- Python 3.9+ (uses `list[str]` style type hints)
- `ffmpeg` on PATH
- An `OPENAI_API_KEY` in your environment or a `.env` file
- `pip install openai numpy pandas librosa scikit-learn python-dotenv ipython`

## Running it

```bash
export OPENAI_API_KEY=sk-...
python support_call_pipeline.py
```

Drop a short recording (~10–15 seconds, one clear sentence) at
`sample_support_call.m4a` in the working directory. If it's missing,
the script falls back to a scripted transcript so the rest of the
pipeline still runs and demonstrates correctly.

## Configuration

All model names are overridable via env vars, with sane defaults:

| Env var | Default |
|---|---|
| `OPENAI_STT_MODEL` | `gpt-4o-transcribe` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `OPENAI_ANSWER_MODEL` | `gpt-5-mini` |
| `OPENAI_TTS_MODEL` | `tts-1` |
| `OPENAI_TTS_VOICE` | `coral` |

## Why normalize embeddings?

`normalize_rows()` scales every embedding to unit length before
comparison. With unit-length vectors, cosine similarity reduces to a
plain dot product — cheaper to compute, and it means differences in
score reflect *direction* (semantic content) rather than incidental
differences in vector magnitude between texts of different lengths.

See `CAVEATS.md` for known limitations and things to watch out for
before using this as anything more than a demo.
