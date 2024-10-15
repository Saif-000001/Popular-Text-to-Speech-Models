# Architecture: Transformer-based sequence-to-sequence model

# Key Features:
    # - Uses self-attention mechanisms instead of RNNs
    # - Can capture long-range dependencies in text
    # - Often faster to train than RNN-based models
    # - Produces mel spectrograms as output

from TTS.api import TTS
from IPython.display import Audio

# Initialize TTS with Transformer TTS
tts = TTS(model_name="tts_models/en/ljspeech/transformer")

# Generate speech and get the waveform directly
text = "Hello, this is Transformer TTS speaking."
speech = tts.tts(text)

# Play the generated speech (if running in a Jupyter or IPython environment)
Audio(speech, rate=tts.synthesizer.output_sample_rate)
