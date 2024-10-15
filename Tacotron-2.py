# Architecture: Sequence-to-sequence model with attention mechanism


# Key Features:
    # - Uses a recurrent neural network (RNN) encoder-decoder
    # - Incorporates an attention mechanism
    # - Produces mel spectrograms as output
    # - Often paired with WaveNet as a vocoder

from TTS.api import TTS
from IPython.display import Audio

# Initialize TTS with Tacotron 2
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Generate speech and get the waveform directly
text = "Hello, this is Tacotron 2 speaking."
speech = tts.tts(text)

# Play the generated speech (if running in a Jupyter or IPython environment)
Audio(speech, rate=tts.synthesizer.output_sample_rate)
