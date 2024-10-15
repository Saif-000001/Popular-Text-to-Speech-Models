# Architecture: Transformer-based sequence-to-sequence model

# Key Features:
    # - Uses self-attention mechanisms instead of RNNs
    # - Can capture long-range dependencies in text
    # - Often faster to train than RNN-based models
    # - Produces mel spectrograms as output

from TTS.api import TTS

# Initialize TTS with Transformer TTS
tts = TTS(model_name="tts_models/en/ljspeech/transformer")

# Generate speech
tts.tts_to_file(text="Hello, this is Transformer TTS speaking.", file_path="output.wav")