import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DModel(nn.Module):
    def __init__(self, num_features, dropout_p = 0, seq_length = 1000, dtype = torch.float):
        super(Conv1DModel, self).__init__()
        # 1: noisy data
        self.num_features = num_features
        self.seq_length = seq_length
        self.conv1 = nn.Conv1d(in_channels = self.num_features, out_channels = 28, kernel_size = 7, stride = 1, padding = 3).to(dtype)
        self.conv2 = nn.Conv1d(in_channels = 28, out_channels = 48, kernel_size = 5, stride = 1, padding = 2).to(dtype)
        self.conv3 = nn.Conv1d(in_channels = 48, out_channels = 84, kernel_size = 3, stride = 1, padding = 1).to(dtype)
        self.dropout = nn.Dropout(dropout_p).to(dtype)
        self.fc1 = nn.Linear(84 * self.seq_length, 64).to(dtype)
        self.fc2 = nn.Linear(64, self.seq_length).to(dtype)
    
    def forward(self, x):
        x = torch.tensor(x.clone().detach().requires_grad_(True), dtype=self.conv1.weight.dtype)
        x = x.unsqueeze(1)
        out = F.silu(self.conv1(x))
        out = F.silu(self.conv2(out))
        out = F.silu(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = F.silu(self.fc1(self.dropout(out)))
        out = self.fc2(self.dropout(out))
        return out