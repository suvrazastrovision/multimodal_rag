# Caveats & Limitations

This is a **teaching demo**, not a production pipeline. Known issues
and things to fix before using it for anything real:

## Retrieval / embeddings

- **`past_embeddings` is recomputed on every call.** The archive
  (`PAST_CALLS`) never changes between requests, so re-embedding it
  each time is wasted API calls and latency. In production: precompute
  once, cache to disk/DB, or use a proper vector store (pgvector,
  Pinecone, Qdrant, Weaviate) with an index — only the *new* query
  needs a live embedding call.
- **No real vector search.** `cosine_similarity` over a small in-memory
  NumPy array is fine for 4 rows; it does not scale to a real call
  archive. A brute-force scan like this becomes the bottleneck well
  before thousands of rows.
- **`top_k=2` is hardcoded implicitly via the demo dataset size.** With
  a larger archive you'd want to tune `top_k`, add a similarity
  threshold (don't return low-confidence matches), and consider
  re-ranking rather than pure cosine similarity.

## ASR / transcription

- **No retry/error handling.** A transient API failure or malformed
  audio file will raise an unhandled exception. Production code needs
  retries with backoff and a clear failure path.
- **Transcription errors propagate silently.** The pipeline has no way
  to flag low-confidence transcriptions — a misheard word (e.g.
  "reset" → "rest") will quietly produce a wrong retrieval match and a
  wrong LLM answer, with nothing in the output signaling that anything
  went wrong.
- **`FALLBACK_TRANSCRIPT` is a demo convenience, not error handling.**
  It exists so the pipeline still runs without a real recording. Don't
  mistake it for a real "no audio detected" code path.

## LLM answer generation

- **Grounding depends entirely on the system prompt**, not on any
  structural guarantee. "Use only the transcript and retrieved
  context" is an instruction, not an enforced constraint — the model
  can still hallucinate. There's no output validation checking the
  answer against the retrieved context.
- **No conversation memory / multi-turn handling.** Each call is
  stateless; this doesn't model a real support conversation with
  back-and-forth.
- **No PII handling.** Real support transcripts often contain names,
  emails, account numbers. This demo doesn't redact or handle any of
  that — don't point it at real customer data as-is.

## TTS

- **`clean_for_tts` is a blunt regex filter.** It strips anything
  outside a fixed character set, which can mangle legitimate content
  (e.g. currency symbols, non-ASCII names, em dashes). Fine for a
  short demo answer; not robust for arbitrary LLM output.
- **No handling of TTS length limits.** Very long LLM answers could
  exceed API input limits; there's no chunking.

## General

- **No tests.** This is a linear script meant to be read top to
  bottom, not a tested library.
- **No logging, only `print()`.** Fine for a live demo in a notebook,
  not fine for anything you'd run unattended.
- **Costs real API credits.** Every run makes at least 3 paid API
  calls (embedding archive + query, LLM answer, TTS) plus one ASR call
  if a real recording is provided. Don't loop this in CI without
  thinking about cost.
