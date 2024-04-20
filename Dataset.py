from torch.utils.data import Dataset
import random
import numpy as np
import torch
from scipy import signal

class FeatureDataset(Dataset):
    def __init__(self, sequence, dtype = torch.float64, seq_length = 28, transforms = None):
        self.sequence = sequence
        self.dtype = dtype
        self.seq_length = seq_length
        self.transforms = transforms
        
        # generate noise parameters
        self.fs = 125000
        self.emg_freq_band = (50, 150)
        self.

    def __len__(self):
        return self.sequence.__len__() - self.seq_length + 1
    
    def __getitem__(self, index):
        sample = self.sequence[index: index + self.seq_length]
        noisy_sample = 
        output = torch.Tensor(sample)
        if self.transforms != None:
            output = self.transforms(output)
        return output.to(self.dtype)
    
    def generate_noisy_signal(self, signal, fs, freq_band, duration):
        t = np.arange(0, duration, 1/fs)
        noise = np.random.normal(0, 1, len(t))
        sos = signal.butter(10, freq_band, 'bandpass', fs=fs, output='sos')
        filtered_noise = signal.sosfilt(sos, noise)
        noisy_signal = signal + filtered_noise
        return noisy_signal