# VITS (Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech)

# Architecture: End-to-end model combining TTS and vocoder

# Key Features:
    # - Uses a conditional variational autoencoder
    # - Incorporates adversarial training
    # - Produces high-quality speech directly from text
    # - Can generate diverse speech styles

import torch
from TTS.api import TTS
from IPython.display import Audio

# Initialize TTS with VITS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/en/vctk/vits", progress_bar=False).to(device)

# Check available speakers
print("Available speakers:", tts.speakers)

# Generate speech
speaker = tts.speakers[0]  # Select the first available speaker
file_path = "output.wav"  # Specify the output file path
tts.tts_to_file(text="Hello, this is VITS speaking.", file_path=file_path, speaker=speaker)

# Display and play the generated audio
Audio(file_path)
