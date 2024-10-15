import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
from IPython.display import Audio, display

# Load pre-trained models
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load xvector containing speaker's voice characteristics
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Synthesize speech
text = "Hello, this is a test of text-to-speech synthesis using a pre-trained model."
inputs = processor(text=text, return_tensors="pt")

speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)

# Save the audio
sf.write("output.wav", speech.numpy(), samplerate=16000)

# Display the audio
display(Audio("output.wav", autoplay=True))

print("Audio saved as 'output.wav' and should be playing now.")