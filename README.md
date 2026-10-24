# Mount Everest Soundpiece

Transcribe a Mount Everest audio recording and generate a concise, transcript-grounded summary.

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
