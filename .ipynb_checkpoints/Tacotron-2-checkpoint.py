# Architecture: Sequence-to-sequence model with attention mechanism


# Key Features:
    # - Uses a recurrent neural network (RNN) encoder-decoder
    # - Incorporates an attention mechanism
    # - Produces mel spectrograms as output
    # - Often paired with WaveNet as a vocoder

from TTS.api import TTS

# Initialize TTS with Tacotron 2
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Generate speech
tts.tts_to_file(text="Hello, this is Tacotron 2 speaking.", file_path="output.wav")