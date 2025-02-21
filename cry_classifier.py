import os
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import threading  # Import threading for non-blocking input

class BabyCryDataset(Dataset):
    def __init__(self, data_dir, transform=None, fixed_length=300, device='cuda'): # Added device argument
        self.data_dir = data_dir
        self.classes = os.listdir(data_dir)
        self.file_paths = []
        self.labels = []
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            for file_name in os.listdir(class_dir):
                self.file_paths.append(os.path.join(class_dir, file_name))
                self.labels.append(label)
        self.transform = transform
        self.fixed_length = fixed_length #Added fixed length
        self.device = device # Added device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio, sr = librosa.load(self.file_paths[idx], sr=16000)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram = torch.tensor(mel_spectrogram).float()
        mel_spectrogram = mel_spectrogram.to(self.device) # Move mel_spectrogram to device here

        # Padding/Truncating
        if mel_spectrogram.shape[1] < self.fixed_length:
            pad_size = self.fixed_length - mel_spectrogram.shape[1]
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, pad_size))
        elif mel_spectrogram.shape[1] > self.fixed_length:
            mel_spectrogram = mel_spectrogram[:, :self.fixed_length]

        label = torch.tensor(self.labels[idx]).long()
        label = label.to(self.device) # Move label to device here
        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)
        return mel_spectrogram, label

class CryClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CryClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 64 * 150, num_classes) # Corrected input size

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        print(f"Shape before fc1: {x.shape}") # Debugging print
        x = self.fc1(x)
        return x

def record_audio_chunk(chunk_duration=1, fs=22050):
    """Records audio from the microphone for a short chunk."""
    recording = sd.rec(int(chunk_duration * fs), samplerate=fs, channels=1, blocking=True) # Mono, blocking=True for chunking
    return recording, fs

def classify_cry_chunk(audio_chunk, model, dataset_classes, device, fixed_length=300, fs=22050):
    """
    Classifies baby cry from an audio chunk.

    Args:
        audio_chunk (np.ndarray): Audio chunk as a NumPy array.
        model (nn.Module): Trained CryClassifier model.
        dataset_classes (list): List of class names from the training dataset.
        device (torch.device): Device to run inference on (CPU or GPU).
        fixed_length (int): Fixed length for mel spectrogram padding/truncation.
        fs (int): Sampling rate.

    Returns:
        tuple: Predicted cry class and confidence score (as a string percentage).
    """
    audio_trimmed, index = librosa.effects.trim(audio_chunk, top_db=20) # Adjust top_db as needed
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_trimmed.flatten(), sr=fs, n_mels=128) # Flatten if stereo
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = torch.tensor(mel_spectrogram).float().unsqueeze(0).to(device) # Add batch dimension and move to device

    # Padding/Truncating for classification
    if mel_spectrogram.shape[2] < fixed_length: # Corrected dimension index
        pad_size = fixed_length - mel_spectrogram.shape[2] # Corrected dimension index
        mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, pad_size), 'constant', 0) # Corrected padding dimension
    elif mel_spectrogram.shape[2] > fixed_length: # Corrected dimension index
        mel_spectrogram = mel_spectrogram[:, :, :fixed_length] # Corrected slicing dimension

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        output = model(mel_spectrogram)
        probabilities = torch.softmax(output, dim=1) # Get probabilities
        confidence, predicted_class_idx = torch.max(probabilities, 1)
        predicted_class = dataset_classes[predicted_class_idx]
        confidence_percentage = f"{confidence.item() * 100:.2f}%" # Format confidence as percentage
    return predicted_class, confidence_percentage

def input_thread(stop_event):
    """Thread to listen for 'exit' input."""
    while not stop_event.is_set():
        user_input = input()
        if user_input.lower() == 'exit':
            stop_event.set() # Signal to stop recording and classification
            break

if __name__ == '__main__':
    # --- Re-initialize dataset and model for classification ---
    data_dir = 'donateacry-corpus/donateacry_corpus_cleaned_and_updated_data/' # Ensure this is the correct path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BabyCryDataset(data_dir, device=device)
    num_classes = len(dataset.classes)
    model = CryClassifier(num_classes=num_classes).to(device)

    # --- Load Pre-trained Model Weights ---
    model_path = 'baby_cry_model_torch.pth'  # Path to your saved model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device)) # Load weights
        print(f"Loaded pre-trained model weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {model_path}. Please train the model and save the weights first, or place the pre-trained weights file at the specified path.")
        exit()

    # --- Continuous Recording and Classification Loop ---
    chunk_duration = 2  # Classify every 1 second of audio
    stop_event = threading.Event() # Event to stop threads
    input_listener = threading.Thread(target=input_thread, args=(stop_event,))
    input_listener.daemon = True # Allow main thread to exit even if input_listener is running
    input_listener.start()

    print("Continuous baby cry classification started. Type 'exit' to quit.")

    try:
        while not stop_event.is_set():
            audio_chunk, fs = record_audio_chunk(chunk_duration=chunk_duration)
            predicted_cry, confidence = classify_cry_chunk(audio_chunk, model, dataset.classes, device, fs=fs)
            print(f"Predicted Cry Class: {predicted_cry}, Confidence: {confidence}")

    except KeyboardInterrupt: # Handle Ctrl+C if needed as a backup
        print("\nContinuous classification interrupted by user.")
    finally:
        stop_event.set() # Ensure threads are stopped cleanly
        input_listener.join(timeout=1) # Wait for input thread to finish, with timeout

    print("Exiting Continuous Baby Cry Classifier.")