import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
from IPython.display import Audio, display
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from sklearn.metrics import accuracy_score
import numpy as np

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

# Generate speech
speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)

# Save the audio
sf.write("output.wav", speech.numpy(), samplerate=16000)

# Display the audio
display(Audio("output.wav", autoplay=True))

print("Audio saved as 'output.wav' and should be playing now.")

# Step 2: Load ASR Model for Transcription
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Transcribe the generated audio
audio_input, _ = sf.read("output.wav")
inputs = asr_processor(audio_input, return_tensors="pt", padding="longest")
with torch.no_grad():
    logits = asr_model(inputs.input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)

# Decode the predicted ids to text
transcribed_text = asr_processor.batch_decode(predicted_ids)[0]
print(f"Transcribed Text: {transcribed_text}")

# Step 3: Calculate Accuracy
# Lowercase both original and transcribed texts for case-insensitive comparison
original_words = text.lower().split()
transcribed_words = transcribed_text.lower().split()

# Calculate accuracy as the ratio of correct words
correct_words = sum(1 for word in transcribed_words if word in original_words)
accuracy = correct_words / len(original_words) * 100 if original_words else 0

print(f"Original Text: '{text}'")
print(f"Transcribed Text: '{transcribed_text}'")
print(f"Accuracy: {accuracy:.2f}%")
