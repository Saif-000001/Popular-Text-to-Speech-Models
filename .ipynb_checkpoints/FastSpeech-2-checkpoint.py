# Architecture: Non-autoregressive Transformer-based model

# Key Features:
    # - Faster than autoregressive models like Tacotron 2
    # - Uses a feed-forward Transformer network
    # - Incorporates duration, pitch, and energy predictors
    # - Can control speaking rate and pitch

from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none

# Initialize FastSpeech 2 model
model = Text2Speech.from_pretrained(model_tag="kan-bayashi/ljspeech_fastspeech2")

# Generate speech
speech = model("Hello, this is FastSpeech 2 speaking.")