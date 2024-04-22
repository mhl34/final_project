import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import numpy as np
import random
import datetime
import sys
import os
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F
from EcgDataset import EcgDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from Conv1DModel import Conv1DModel
from TransformerModel import TransformerModel
import matplotlib.pyplot as plt


class runModel:
    def __init__(self, mainDir):
        # device parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {self.device}")

        # file parameters
        self.ecg_data_file = "resampled_Matthew_Normal_0.csv"

        # dataloader parameters
        self.ecg_sequence, self.time_sequence = self.process_data(self.ecg_data_file)
        self.dtype = torch.float64
        self.seq_length = 1000
        self.num_epochs = 20
        self.batch_size = 32
        
        # optimizer parameters
        self.lr = 1e-3
        self.weight_decay = 1e-8

        # model parameters
        self.model_type = "transformer"
        self.num_features = 1
        self.dropout_p = 0.5
        
        # transformer parameters
        self.dim_model = 1024
        self.num_head = 128

    def train(self, model):
        print(self.device)
        print("============================")
        print("Training...")
        print("============================")
        model.train()

        train_dataset = EcgDataset(self.ecg_sequence, self.time_sequence, dtype = self.dtype, seq_length = self.seq_length, transforms = None)
        train_dataloader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)

        for epoch in range(self.num_epochs):

            progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')
            
            lossLst = []
            accLst = []
            
            for batch_idx, (data) in progress_bar:
                input, target, _ = data
                input = input.to(self.device)
                target = target.to(self.device)
                
                if self.model_type == "conv1d":
                    output = model(input).to(self.dtype)
                elif self.model_type == "transformer":
                    output = model(target, input).to(self.dtype).squeeze()
                
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward(retain_graph = True)
                optimizer.step()

                lossLst.append(loss.item())
                accLst.append(1 - self.mape(output, target))
                # persAccList.append(self.persAcc(output, target))
            scheduler.step()

            print(f"epoch {epoch + 1} training loss: {sum(lossLst)/len(lossLst)} learning rate: {scheduler.get_last_lr()} training accuracy: {sum(accLst)/len(accLst)}")
            
            print(output.shape, target.shape)

            # print(f"pers category accuracy: {sum(persAccList)/len(persAccList)}")

            # example output with the epoch
            for outVal, targetVal in zip(output[-1][:3], target[-1][:3]):
                print(f"output: {outVal.item()}, target: {targetVal.item()}, difference: {outVal.item() - targetVal.item()}")

            plt.clf()
            plt.grid(True)
            plt.figure(figsize=(8, 6))

            # Plot the target array
            plt.plot(target.detach().cpu().numpy()[-1], label='Target')

            # Plot the output arrays (first and second arrays in the tuple)
            plt.plot(output.detach().cpu().numpy()[-1], label='Output')
            
            plt.plot(input.detach().cpu().numpy()[-1], label='Noisy Input')

            # Add labels and legend
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title(f'Target vs. Output', fontdict={'fontsize': 16, 'fontweight': 'bold'})
            plt.legend()

            plt.savefig(f'plots/{self.model_type}_{epoch}_output.png')
        
                    
    def evaluate(self, model):
        print("============================")
        print("Evaluating...")
        print("============================")
        model.eval()

        val_dataset = EcgDataset(self.ecg_sequence, self.time_sequence, dtype = self.dtype, seq_length = self.seq_length, transforms = None)
        # returns eda, hr, temp, then hba1c
        val_dataloader = DataLoader(val_dataset, batch_size = self.batch_size, shuffle = False)

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')

            lossLst = []
            accLst = []


            for batch_idx, (data) in progress_bar:
                input, target, _ = data
                input = input.to(self.device)
                target = target.to(self.device)

                if self.model_type == "conv1d":
                    output = model(input).to(self.dtype)
                elif self.model_type == "transformer":
                    output = model(target, input).to(self.dtype).squeeze()

                loss = criterion(output, target)

                loss = criterion(output, target)

                lossLst.append(loss.item())
                accLst.append(1 - self.mape(output, target))
                # persAccList.append(self.persAcc(output, glucStats))

            print(f"epoch {epoch} training loss: {sum(lossLst)/len(lossLst)} training accuracy: {sum(accLst)/len(accLst)}")

            # print(f"pers category accuracy: {sum(persAccList)/len(persAccList)}")

            # example output with the final epoch
            for outVal, targetVal in zip(output[-1][:3], target[-1][:3]):
                    print(f"output: {outVal.item()}, target: {targetVal.item()}, difference: {outVal.item() - targetVal.item()}")
                

    def mape(self, pred, target):
        return (torch.mean(torch.div(torch.abs(target - pred), torch.abs(target)))).item()

    def process_data(self, data_file):
        df = pd.read_csv(data_file)
        return df['V'].to_numpy(), df['Unit'].to_numpy()

    def run(self):
        # model = Conv1DModel(self.num_features, dropout_p = self.dropout_p, seq_length = self.seq_length, dtype = self.dtype).to(self.device)
        model = TransformerModel(num_features = self.dim_model, num_head = self.num_head, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, device = self.device, dtype = self.dtype, num_seqs = self.num_features).to(self.device)
        self.train(model)
        self.evaluate(model)

if __name__ == "__main__":
    mainDir = "/media/nvme1/expansion/glycemic_health_data/physionet.org/files/big-ideas-glycemic-wearable/1.1.2/"
    obj = runModel(mainDir)
    obj.run()