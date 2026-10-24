"""
Mount Everest soundpiece pipeline.

Pipeline steps:
  1. Setup & verification    - check env vars, ffmpeg, input file
  2. Audio preprocessing     - convert to 16kHz mono WAV
  3. Transcription (ASR)     - speech-to-text via OpenAI
  4. Summary generation      - summarize the soundpiece and its themes
  5. Text-to-speech (TTS)    - spoken output of the summary
"""

from pathlib import Path
import os
import re
import shutil
import subprocess

import librosa
from dotenv import load_dotenv
from IPython.display import Audio, display
from openai import OpenAI

load_dotenv()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

INPUT_AUDIO = Path("mount_everest.m4a")
CLEAN_AUDIO = Path("clean_audio_16k_mono.wav")
TRANSCRIPT_OUTPUT = Path("mount_everest_transcript.txt")
SUMMARY_OUTPUT = Path("mount_everest_summary.txt")
TTS_OUTPUT = Path("mount_everest_summary.wav")

STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe")
ANSWER_MODEL = os.getenv("OPENAI_ANSWER_MODEL", "gpt-5-mini")
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "shimmer")
GENERATE_SPOKEN_SUMMARY = os.getenv(
    "GENERATE_SPOKEN_SUMMARY", "false"
).lower() in {"1", "true", "yes"}


# --------------------------------------------------------------------------
# Step 1: Setup and verification
# --------------------------------------------------------------------------

def verify_environment() -> None:
    """Check that required env vars, tooling, and input files are present."""
    print("OpenAI key set:", bool(os.environ.get("OPENAI_API_KEY")))
    print("FFmpeg available:", _find_ffmpeg() is not None)
    print("Input audio exists:", INPUT_AUDIO.exists())
    print("Answer model:", ANSWER_MODEL)


def _find_ffmpeg() -> str | None:
    """Find FFmpeg on PATH or in a standard Windows Winget installation."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        for package in sorted(packages.glob("Gyan.FFmpeg*")):
            candidates = sorted(package.glob("**/bin/ffmpeg.exe"))
            if candidates:
                return str(candidates[0])

    return None


# --------------------------------------------------------------------------
# Step 2: Audio preprocessing
# --------------------------------------------------------------------------

def preprocess_audio(input_path: Path, output_path: Path) -> None:
    """Convert input audio to 16kHz mono WAV using ffmpeg, then report stats."""
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise FileNotFoundError(
            "FFmpeg was not found. Install it and add it to PATH, then restart VS Code."
        )

    cmd = [
        ffmpeg,
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
            prompt=(
                "A Mount Everest soundpiece. Preserve names and terms such as "
                "Everest, Khumbu, Himalaya, Sherpa, and summit when spoken."
            ),
        )
    return transcription.text.strip()


def get_transcript(client: OpenAI) -> str:
    """Preprocess and transcribe the supplied soundpiece."""
    if not INPUT_AUDIO.exists():
        raise FileNotFoundError(
            f"Audio file not found: {INPUT_AUDIO}. "
            "Set INPUT_AUDIO to the path of your Mount Everest soundpiece."
        )

    preprocess_audio(INPUT_AUDIO, CLEAN_AUDIO)
    transcript = transcribe_audio(client, CLEAN_AUDIO)
    print(transcript)
    return transcript


# --------------------------------------------------------------------------
# Step 4: Summary generation (LLM)
# --------------------------------------------------------------------------

def generate_summary(client: OpenAI, transcript: str) -> str:
    """Summarize the soundpiece and identify themes using its transcript."""
    user_prompt = (
        "Transcript of a Mount Everest soundpiece:\n"
        f"{transcript}\n\n"
        "Task:\n"
        "1. Give a concise summary of what the soundpiece communicates.\n"
        "2. Identify its central themes and emotional tone.\n"
        "3. Mention notable people, places, or events only if they appear "
        "in the transcript.\n"
        "Keep it clear and suitable as a short documentary-style narration."
    )

    response = client.responses.create(
        model=ANSWER_MODEL,
        input=[
            {
                "role": "developer",
                "content": (
                    "Your name is Suvra. You are an attentive and experienced nature photographer. Base your "
                    "summary only on the supplied transcript. Do not invent "
                    "audio details, historical facts, people, or events. "
                    "A transcript cannot establish what music, ambience, or "
                    "other nonverbal sounds are present."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )

    summary = response.output_text.strip()
    print(summary)
    return summary


# --------------------------------------------------------------------------
# Step 5: Text-to-speech (TTS)
# --------------------------------------------------------------------------

def clean_for_tts(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?;:()'\" ]+", "", text)
    return text.strip()


def synthesize_speech(client: OpenAI, text: str, output_path: Path) -> None:
    """Read the summary aloud and save it to output_path."""
    spoken_answer = clean_for_tts(text)
    print(spoken_answer)

    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=spoken_answer,
        instructions="Speak clearly and reflectively as a documentary narrator.",
        response_format="wav",
    ) as response:
        response.stream_to_file(output_path)

    print("Saved spoken answer to", output_path)
    display(Audio(str(output_path), autoplay=False))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main() -> None:
    client = OpenAI()

    verify_environment()

    transcript = get_transcript(client)
    TRANSCRIPT_OUTPUT.write_text(transcript + "\n", encoding="utf-8")
    print("Saved transcript to", TRANSCRIPT_OUTPUT)

    summary = generate_summary(client, transcript)
    SUMMARY_OUTPUT.write_text(summary + "\n", encoding="utf-8")
    print("Saved summary to", SUMMARY_OUTPUT)

    if GENERATE_SPOKEN_SUMMARY:
        synthesize_speech(client, summary, TTS_OUTPUT)


if __name__ == "__main__":
    main()
