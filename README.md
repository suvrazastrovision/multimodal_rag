[![OpenAI API](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)](https://openai.com/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)

# Mount Everest Soundpiece

Transcribe a Mount Everest audio recording and generate a concise, transcript-grounded summary.

## Models

The pipeline uses these OpenAI models by default:

| Task | Model | When it runs |
| --- | --- | --- |
| Speech-to-text (transcription) | `gpt-4o-transcribe` | Every run |
| Transcript summary | `gpt-5-mini` | Every run |
| Text-to-speech (spoken summary) | `tts-1` (voice: `shimmer`) | When `GENERATE_SPOKEN_SUMMARY=true` |

Override the defaults with `OPENAI_STT_MODEL`, `OPENAI_ANSWER_MODEL`, and `OPENAI_TTS_MODEL` in `.env`. Set `OPENAI_TTS_VOICE` to change the speech voice.

## Run on Windows

1. Install FFmpeg and create a `.env` file containing `OPENAI_API_KEY=your_key`.
2. Put the recording in the project folder as `sample_support_call.m4a`.
3. In PowerShell, install dependencies and run:

   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe .\mount_everest.py
   ```

The OpenAI project must have access to the configured transcription and summary models. Results are saved to `mount_everest_transcript.txt` and `mount_everest_summary.txt`. To also generate spoken audio, set `GENERATE_SPOKEN_SUMMARY=true` in `.env`.
