import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import math
import numpy as np
import random
import datetime
import sys
import os
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class TransformerNoDecodeModel(nn.Module):
    # Constructor
    def __init__(
        self,
        num_features,
        num_head,
        seq_length,
        dropout_p,
        norm_first,
        dtype,
        device,
        num_seqs = 5):
        super(TransformerNoDecodeModel, self).__init__()

        # INFO
        self.model_type = "Transformer"
        self.num_features = num_features
        self.num_head = num_head
        self.seq_length = seq_length
        self.dropout_p = dropout_p
        self.norm_first = norm_first
        self.device = device
        self.dtype = dtype
        self.num_seqs = num_seqs

        # EMBEDDING LINEAR LAYERS
        self.embedding_layer = nn.Linear(self.seq_length, self.num_features, dtype = self.dtype)

        # ENCODER LAYERS
        self.encoder = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = self.norm_first, dtype = self.dtype)

        # DECODER LAYERS
        # self.decoder = nn.TransformerDecoderLayer(d_model=self.num_features, nhead=self.num_head, dtype = self.dtype)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)  # Using a single layer

        # FULLY-CONNECTED LAYERS
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(self.num_features * self.num_seqs, self.num_features, dtype = self.dtype)
        self.fc2 = nn.Linear(self.num_features, self.num_features // 2, dtype = self.dtype)
        self.fc3 = nn.Linear(self.num_features // 2, self.seq_length, dtype = self.dtype)

    # function: forward of model
    # input: src, tgt, tgt_mask
    # output: output after forward run through model
    # def forward(self, tgt, src):
    # tgt = self.embedding_layer(tgt).unsqueeze(1)
        
    # output = self.decoder(tgt = tgt, memory = output, tgt_mask = self.get_tgt_mask(len(tgt)))
    
    def forward(self, src):
        output = self.embedding_layer(src).unsqueeze(1)

        output = self.encoder(output).to(self.dtype)
        
        output = F.silu(self.fc1(output))
        
        output = torch.tensor(output.clone().detach().requires_grad_(True), dtype=self.fc1.weight.dtype)
        
        output = F.silu(self.fc2(self.dropout(output)))
        output = self.fc3(self.dropout(output))
        return output
    
    # function: creates a mask with 0's in bottom left of matrix
    # input: size
    # output: mask
    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size,size) * float('-inf')).T
        for i in range(size):
            mask[i, i] = 0
        return mask.to(self.dtype).to(self.device)
    
class TransformerModel(nn.Module):
    # Constructor
    def __init__(
        self,
        num_features,
        num_head,
        seq_length,
        dropout_p,
        norm_first,
        dtype,
        device,
        num_seqs = 5):
        super(TransformerModel, self).__init__()

        # INFO
        self.model_type = "Transformer"
        self.num_features = num_features
        self.num_head = num_head
        self.seq_length = seq_length
        self.dropout_p = dropout_p
        self.norm_first = norm_first
        self.device = device
        self.dtype = dtype
        self.num_seqs = num_seqs

        # EMBEDDING LINEAR LAYERS
        self.embedding_layer = nn.Linear(self.seq_length, self.num_features, dtype = self.dtype)

        # ENCODER LAYERS
        self.encoder = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = self.norm_first, dtype = self.dtype)

        # DECODER LAYERS
        # self.decoder = nn.TransformerDecoderLayer(d_model=self.num_features, nhead=self.num_head, dtype = self.dtype)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)  # Using a single layer

        # FULLY-CONNECTED LAYERS
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(self.num_features * self.num_seqs, self.num_features, dtype = self.dtype)
        self.fc2 = nn.Linear(self.num_features, self.num_features // 2, dtype = self.dtype)
        self.fc3 = nn.Linear(self.num_features // 2, self.seq_length, dtype = self.dtype)

    # function: forward of model
    # input: src, tgt, tgt_mask
    # output: output after forward run through model
    def forward(self, tgt, src):
        output = self.embedding_layer(src).unsqueeze(1)

        output = self.encoder(output).to(self.dtype)
        
        output = F.silu(self.fc1(output))
        
        tgt = self.embedding_layer(tgt).unsqueeze(1)
        
        output = self.decoder(tgt = tgt, memory = output, tgt_mask = self.get_tgt_mask(len(tgt)))
    
        
        output = torch.tensor(output.clone().detach().requires_grad_(True), dtype=self.fc1.weight.dtype)
        
        output = F.silu(self.fc2(self.dropout(output)))
        output = self.fc3(self.dropout(output))
        return output
    
    # function: creates a mask with 0's in bottom left of matrix
    # input: size
    # output: mask
    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size,size) * float('-inf')).T
        for i in range(size):
            mask[i, i] = 0
        return mask.to(self.dtype).to(self.device)
