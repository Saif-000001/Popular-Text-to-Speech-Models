import torch
import torch.nn as nn
import torchaudio
from transformers import AutoTokenizer, AutoModel

class TTSModel(nn.Module):
    def __init__(self):
        super(TTSModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.decoder = nn.GRU(768, 256, batch_first=True)
        self.mel_proj = nn.Linear(256, 80)  # Assuming 80 mel bins

    def forward(self, text):
        encoded = self.text_encoder(text)[0]
        decoded, _ = self.decoder(encoded)
        mel = self.mel_proj(decoded)
        return mel

def text_to_speech(text, model, tokenizer, vocoder):
    # Tokenize input text
    tokens = tokenizer(text, return_tensors="pt")
    
    # Generate mel spectrogram
    with torch.no_grad():
        mel = model(tokens.input_ids)
    
    # Convert mel spectrogram to audio
    audio = vocoder(mel)
    
    return audio

# Load pre-trained models
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tts_model = TTSModel()
tts_model.load_state_dict(torch.load("path_to_your_pretrained_model.pth"))
vocoder = torchaudio.transforms.GriffinLim(n_fft=1024, n_iter=60)

# Predict
text = "Hello, this is a test of text-to-speech synthesis."
audio = text_to_speech(text, tts_model, tokenizer, vocoder)

# Save audio
torchaudio.save("output.wav", audio, sample_rate=22050)