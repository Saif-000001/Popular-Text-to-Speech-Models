# Architecture: Non-autoregressive Transformer-based model

# Key Features:
    # - Faster than autoregressive models like Tacotron 2
    # - Uses a feed-forward Transformer network
    # - Incorporates duration, pitch, and energy predictors
    # - Can control speaking rate and pitch

from espnet2.bin.tts_inference import Text2Speech
from IPython.display import Audio

# Initialize FastSpeech 2 model
model = Text2Speech.from_pretrained(model_tag="kan-bayashi/ljspeech_fastspeech2")

# Generate speech
text = "Hello, this is FastSpeech 2 speaking."
output = model(text)

# Extract the waveform (usually stored under 'wav')
speech = output['wav']

# Play the generated speech (if running in a Jupyter or IPython environment)
Audio(speech, rate=model.fs)
