from torch.utils.data import Dataset
import random
import numpy as np
import torch
from scipy import signal

class EcgDataset(Dataset):
    def __init__(self, ecg_sequence, time_sequence, dtype = torch.float64, seq_length = 28, transforms = None):
        self.ecg_sequence = ecg_sequence
        self.time_sequence = time_sequence
        self.dtype = dtype
        self.seq_length = seq_length
        self.transforms = transforms
        
        # generate noise parameters
        self.fs = 125000
        self.emg_freq_band = (50, 150)
        self.duration = time_sequence[-1]

    def __len__(self):
        return self.ecg_sequence.__len__() - self.seq_length + 1
    
    def __getitem__(self, index):
        sample = self.ecg_sequence[index: index + self.seq_length]
        noisy_sample = self.generate_noisy_data(self.ecg_sequence, self.fs, self.time_sequence, self.emg_freq_band, self.duration)
        time_sample = self.time_sequence[index: index + self.seq_length]

        sample = torch.Tensor(sample)
        noisy_sample = torch.Tensor(noisy_sample)
        time_sample = torch.Tensor(time_sample)

        if self.transforms != None:
            sample = self.transforms(sample)
            noisy_sample = self.transforms(sample)
        return noisy_sample.to(self.dtype), sample.to(self.dtype), time_sample.to(self.dtype)
    
    def generate_noisy_data(self, data, fs, time, freq_band, duration):
        noise = np.random.normal(0, 1, len(time))
        sos =  signal.butter(10, freq_band, 'bandpass', fs=fs, output='sos')
        filtered_noise = signal.sosfilt(sos, noise)
        noisy_data = data + filtered_noise * 5
        return noisy_data