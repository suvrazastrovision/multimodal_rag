"""
Support call pipeline demo.

Pipeline steps:
  1. Setup & verification    - check env vars, ffmpeg, input file
  2. Audio preprocessing     - convert to 16kHz mono WAV
  3. Transcription (ASR)     - speech-to-text via OpenAI
  4. Retrieval (embeddings)  - find similar past calls
  5. Response generation     - LLM answer grounded in transcript + retrieval
  6. Text-to-speech (TTS)    - spoken output of the answer
"""

from pathlib import Path
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import librosa
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import Audio, display
from openai import OpenAI


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

INPUT_AUDIO = Path("sample_support_call.m4a")
CLEAN_AUDIO = Path("clean_audio_16k_mono.wav")
TTS_OUTPUT = Path("answer_openai.wav")

STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
ANSWER_MODEL = os.getenv("OPENAI_ANSWER_MODEL", "gpt-5-mini")
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "coral")

FALLBACK_TRANSCRIPT = (
    "Hi, I cannot log into my account. "
    "The password reset email never arrives, and I need access today."
)

PAST_CALLS = [
    {
        "id": "call_101",
        "topic": "billing",
        "transcript": "The customer was charged twice for the same invoice and wants a refund.",
    },
    {
        "id": "call_102",
        "topic": "password_reset",
        "transcript": "The user cannot log in because the password reset email is not arriving.",
    },
    {
        "id": "call_103",
        "topic": "delivery_delay",
        "transcript": "The customer asks why their package is delayed and wants a new delivery date.",
    },
    {
        "id": "call_104",
        "topic": "audio_issue",
        "transcript": "The microphone is not detected during video calls after a laptop update.",
    },
]


# --------------------------------------------------------------------------
# Step 1: Setup and verification
# --------------------------------------------------------------------------

def verify_environment(client: OpenAI) -> None:
    """Check that required env vars, tooling, and input files are present."""
    print("OpenAI key set:", bool(os.environ.get("OPENAI_API_KEY")))
    print("FFmpeg available:", bool(shutil.which("ffmpeg")))
    print("Input audio exists:", INPUT_AUDIO.exists())
    print("Answer model:", ANSWER_MODEL)


# --------------------------------------------------------------------------
# Step 2: Audio preprocessing
# --------------------------------------------------------------------------

def preprocess_audio(input_path: Path, output_path: Path) -> None:
    """Convert input audio to 16kHz mono WAV using ffmpeg, then report stats."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print("Saved", output_path, "with FFmpeg")

    audio, sample_rate = librosa.load(output_path, sr=16000, mono=True)
    duration = len(audio) / sample_rate

    print("Sample rate:", sample_rate)
    print("Duration:", round(duration, 2), "seconds")
    print("Samples:", audio.shape)

    display(Audio(str(output_path), autoplay=False))


# --------------------------------------------------------------------------
# Step 3: Transcription (ASR)
# --------------------------------------------------------------------------

def transcribe_audio(client: OpenAI, audio_path: Path) -> str:
    """Transcribe audio via OpenAI STT model."""
    with audio_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=audio_file,
            prompt="A short customer support call",
        )
    return transcription.text.strip()


def get_transcript(client: OpenAI) -> str:
    """Return a transcript: from audio if available, else the fallback text."""
    if not INPUT_AUDIO.exists():
        print("Missing", INPUT_AUDIO, "- using fallback transcript:")
        print(FALLBACK_TRANSCRIPT)
        return FALLBACK_TRANSCRIPT

    preprocess_audio(INPUT_AUDIO, CLEAN_AUDIO)
    transcript = transcribe_audio(client, CLEAN_AUDIO)
    print(transcript)
    return transcript


# --------------------------------------------------------------------------
# Step 4: Retrieval (embeddings)
# --------------------------------------------------------------------------

def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    return normalize_rows(vectors)


def retrieve_similar_calls(
    client: OpenAI, transcript: str, past_calls: list[dict], top_k: int = 2
) -> list[dict]:
    """Embed the transcript and past calls, rank by cosine similarity."""
    past_texts = [item["transcript"] for item in past_calls]

    past_embeddings = embed_texts(client, past_texts)
    query_embedding = embed_texts(client, [transcript])

    scores = cosine_similarity(query_embedding, past_embeddings)[0]

    results = pd.DataFrame({
        "id": [item["id"] for item in past_calls],
        "topic": [item["topic"] for item in past_calls],
        "transcript": past_texts,
        "score": scores,
    }).sort_values("score", ascending=False)

    print(results[["id", "topic", "score"]].to_string(index=False))

    return results.head(top_k).to_dict(orient="records")


# --------------------------------------------------------------------------
# Step 5: Response generation (LLM)
# --------------------------------------------------------------------------

def generate_answer(client: OpenAI, transcript: str, retrieved_context: list[dict]) -> str:
    """Ask the LLM to summarize, classify intent, and suggest a next action."""
    user_prompt = (
        "New call transcript:\n"
        f"{transcript}\n\n"
        "Similar past calls:\n"
        f"{retrieved_context}\n\n"
        "Task:\n"
        "1. Summarize the user issue in one sentence.\n"
        "2. Identify the likely intent.\n"
        "3. Suggest the next best support action.\n"
        "Keep the answer short and clear."
    )

    response = client.responses.create(
        model=ANSWER_MODEL,
        input=[
            {
                "role": "developer",
                "content": (
                    "You are a concise support assistant. "
                    "Use only the transcript and retrieved context. "
                    "Do not invent account details, ticket IDs, or policies."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.output_text.strip()
    print(answer)
    return answer


# --------------------------------------------------------------------------
# Step 6: Text-to-speech (TTS)
# --------------------------------------------------------------------------

def clean_for_tts(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?;:()'\" ]+", "", text)
    return text.strip()


def synthesize_speech(client: OpenAI, text: str, output_path: Path) -> None:
    """Convert text to spoken audio and save it to output_path."""
    spoken_answer = clean_for_tts(text)
    print(spoken_answer)

    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=spoken_answer,
        instructions="Speak clearly and calmly as a customer support assistant.",
        response_format="wav",
    ) as response:
        response.stream_to_file(output_path)

    print("Saved spoken answer to", output_path)
    display(Audio(str(output_path), autoplay=False))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    client = OpenAI()

    verify_environment(client)

    transcript = get_transcript(client)
    retrieved_context = retrieve_similar_calls(client, transcript, PAST_CALLS)
    answer = generate_answer(client, transcript, retrieved_context)
    synthesize_speech(client, answer, TTS_OUTPUT)


if __name__ == "__main__":
    main()
