"""Offline hands-free voice characters on local models.

The pipeline, one package per stage:
  stt/   speech to text   (Whisper: mlx on Apple Silicon, faster-whisper on Linux)
  llm/   text to text     (llama.cpp: persona prompt, reply checks, memory)
  tts/   text to speech   (Piper voices, effects, intonation, playback)

cli.py wires them together, conversation.py runs the loop, persona/ holds the
characters and startup.py prints the banner and checks for missing files.
"""

__version__ = "0.1.0"
