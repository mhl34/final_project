from torch.utils.data import Dataset
import random
import numpy as np
import torch
from scipy import signal
import neurokit2 as nk

class EcgDataset(Dataset):
    def __init__(self, ecg_sequence, time_sequence, dtype = torch.float64, seq_length = 28, transforms = None, smooth = True):
        # parameters
        self.dtype = dtype
        self.seq_length = seq_length
        self.transforms = transforms
        self.smooth = smooth
        
        # sequences
        self.ecg_sequence = ecg_sequence
        if self.smooth:
            self.ecg_sequence = self.smooth_fn(self.ecg_sequence, 50)
        # self.ecg_std = 0.14365994671627114
        # self.ecg_min = -0.2725966580944904
        self.time_sequence = time_sequence
        
        # noise parameter
        self.fs = 125000
        self.emg_freq_band = (50, 250)
        self.duration = time_sequence[-1]
        self.noisy_ecg_sequence = self.generate_noisy_data(self.ecg_sequence, self.fs, self.time_sequence, self.emg_freq_band, self.duration)
        
        # peak parameters
        self.p_peaks = 1
        self.q_peaks = 2
        self.s_peaks = 3
        self.t_peaks = 4
        
        _, rpeaks = nk.ecg_peaks(self.ecg_sequence, sampling_rate=1000)
        _, waves_peak = nk.ecg_delineate(self.ecg_sequence, rpeaks, sampling_rate=1000, method="peak")
        
        self.peaks = np.zeros_like(self.ecg_sequence)

        t_peak_locs = [i for i in waves_peak['ECG_T_Peaks'] if isinstance(i, np.int64)]
        # p_peak_locs = [i for i in waves_peak['ECG_P_Peaks'] if isinstance(i, np.int64)]
        # q_peak_locs = [i for i in waves_peak['ECG_Q_Peaks'] if isinstance(i, np.int64)]
        # s_peak_locs = [i for i in waves_peak['ECG_S_Peaks'] if isinstance(i, np.int64)]
        
        self.peaks[t_peak_locs] = self.t_peaks
        # self.peaks[p_peak_locs] = self.p_peaks
        # self.peaks[q_peak_locs] = self.q_peaks
        # self.peaks[s_peak_locs] = self.s_peaks

    def __len__(self):
        return self.ecg_sequence.__len__() - self.seq_length + 1
    
    def __getitem__(self, index):
        # sample = (self.ecg_sequence[index: index + self.seq_length] - self.ecg_min) / self.ecg_std
        # noisy_sample = (self.noisy_ecg_sequence[index: index + self.seq_length] - self.ecg_min) / self.ecg_std
        sample = self.ecg_sequence[index: index + self.seq_length]
        noisy_sample = self.noisy_ecg_sequence[index: index + self.seq_length]
        time_sample = self.time_sequence[index: index + self.seq_length]
        peaks_sample = self.peaks[index:index + self.seq_length]
        
        sample = torch.Tensor(sample)
        noisy_sample = torch.Tensor(noisy_sample)
        time_sample = torch.Tensor(time_sample)
        peaks_sample = torch.Tensor(peaks_sample)
        
        if self.transforms != None:
            sample = self.transforms(sample)
            noisy_sample = self.transforms(sample)
        
        return noisy_sample.to(self.dtype), sample.to(self.dtype), time_sample.to(self.dtype), peaks_sample.to(self.dtype)
    
    def generate_noisy_data(self, data, fs, time, freq_band, duration):
        noise = np.random.normal(0, 1, len(time))
        sos =  signal.butter(10, freq_band, 'bandpass', fs=fs, output='sos')
        filtered_noise = signal.sosfilt(sos, noise)
        noisy_data = data + filtered_noise * 5
        return noisy_data
    
    def smooth_fn(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    def find_positive_integers(arr, index, seq_length):
        shifted_arr = [num - index for num in arr]
        positive_integers = [num for num in shifted_arr if 0 < num < seq_length]
        return positive_integers
