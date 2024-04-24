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
from TransformerModel import TransformerNoDecodeModel
import matplotlib.pyplot as plt


class runModel:
    def __init__(self, mainDir):
        # device parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {self.device}")

        # file parameters
        self.ecg_data_file = "resampled_Matthew_Normal_0.csv"
        self.checkpoint_folder = "/home/jovyan/work/final_project/saved_model"

        # dataloader parameters
        self.ecg_sequence, self.time_sequence = self.process_data(self.ecg_data_file)
        self.dtype = torch.float64
        self.seq_length = 2000
        self.num_epochs = 50
        self.batch_size = 32
        
        # optimizer parameters
        self.lr = 1e-3
        self.weight_decay = 1e-8

        # model parameters
        self.model_type = "transformer_no_decode"
        self.num_features = 1
        self.dropout_p = 0.5
        
        # transformer parameters
        self.dim_model = 2048
        self.num_head = 512
        
        # ecg sequence parameters
        self.ecg_std = 0.14365994671627114
        self.ecg_min = -0.2725966580944904

    def train(self, model):
        print(self.device)
        print("============================")
        print("Training...")
        print("============================")
        model.train()

        train_dataset = EcgDataset(self.ecg_sequence, self.time_sequence, dtype = self.dtype, seq_length = self.seq_length, transforms = None)
        train_dataloader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)

        for epoch in range(self.num_epochs):

            progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')
            
            lossLst = []
            accLst = []
            
            for batch_idx, (data) in progress_bar:
                input, target, _, peaks = data
                input = input.to(self.device)
                target = peaks.to(self.device)
                
                # if self.model_type == "conv1d":
                output = model(input).to(self.dtype).squeeze()
                # elif self.model_type == "transformer":
                #     output = model(target, input).to(self.dtype).squeeze()
                
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward(retain_graph = True)
                optimizer.step()

                lossLst.append(loss.item())
                accLst.append(1 - self.mape(output, target))
                # persAccList.append(self.persAcc(output, target))
            scheduler.step()

            print(f"epoch {epoch + 1} training loss: {sum(lossLst)/len(lossLst)} learning rate: {scheduler.get_last_lr()} training accuracy: {sum(accLst)/len(accLst)}")
            
            target_peaks = sorted(np.nonzero(target[-1].cpu().detach().numpy())[0])
            predicted_peaks = sorted(np.argsort(output[-1].cpu().detach().numpy())[-len(target_peaks):])
            
            print(predicted_peaks)
            print(target_peaks)

            # print(f"pers category accuracy: {sum(persAccList)/len(persAccList)}")

            # example output with the epoch
            # for outVal, targetVal in zip(output[-1][:3], target[-1][:3]):
            #     print(f"output: {outVal.item()}, target: {targetVal.item()}, difference: {outVal.item() - targetVal.item()}")

            plt.clf()
            plt.grid(True)
            plt.figure(figsize=(8, 6))

            # Plot the target array
            
            
            plt.plot(input.detach().cpu().numpy()[-1], label='Noisy Input')
            
            for idx in range(len(predicted_peaks)):
                plt.axvline(x=predicted_peaks[idx], color='r', linestyle='--')  # Vertical line at each x-value
                plt.axvline(x=target_peaks[idx], color='b', linestyle='--')

            # Add labels and legend
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title(f'Target vs. Output', fontdict={'fontsize': 16, 'fontweight': 'bold'})
            plt.legend()

            plt.savefig(f'plots/{self.model_type}_{epoch}_no_decode_output.png')
        
                    
    def evaluate(self, model, CHECKPOINT_FOLDER):
        print("============================")
        print("Evaluating...")
        print("============================")
        model.eval()
        epoch = 0

        val_dataset = EcgDataset(self.ecg_sequence, self.time_sequence, dtype = self.dtype, seq_length = self.seq_length, transforms = None)
        # returns eda, hr, temp, then hba1c
        val_dataloader = DataLoader(val_dataset, batch_size = self.batch_size, shuffle = False)

        criterion = nn.MSELoss()

        with torch.no_grad():
            progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')

            lossLst = []
            accLst = []


            for batch_idx, (data) in progress_bar:
                input, target, _, peaks = data
                input = input.to(self.device)
                target = peaks.to(self.device)

                # if self.model_type == "conv1d":
                output = model(input).to(self.dtype)
                # elif self.model_type == "transformer":
                #     output = model(target, input).to(self.dtype).squeeze()

                loss = criterion(output, target)

                loss = criterion(output, target)

                lossLst.append(loss.item())
                accLst.append(1 - self.mape(output, target))
                # persAccList.append(self.persAcc(output, glucStats))

            print(f"epoch {epoch} training loss: {sum(lossLst)/len(lossLst)} training accuracy: {sum(accLst)/len(accLst)}")

            # print(f"pers category accuracy: {sum(persAccList)/len(persAccList)}")

            # example output with the final epoch
            # for outVal, targetVal in zip(output[-1][:3], target[-1][:3]):
            #         print(f"output: {outVal.item()}, target: {targetVal.item()}, difference: {outVal.item() - targetVal.item()}")
        if not os.path.exists(CHECKPOINT_FOLDER):
            os.makedirs(CHECKPOINT_FOLDER)
        print("Saving ...")
        state = {'state_dict': model.state_dict(),
                 'epoch': epoch,
                 'lr': self.lr}
        torch.save(state, os.path.join(CHECKPOINT_FOLDER, 'transformer_no_decode_peaks.pth'))
                

    def mape(self, pred, target):
        return (torch.mean(torch.div(torch.abs(target - pred), torch.abs(target)))).item()

    def process_data(self, data_file):
        df = pd.read_csv(data_file)
        return df['V'].to_numpy(), df['Unit'].to_numpy()

    def run(self):
        if self.model_type == "conv1d":
            model = Conv1DModel(self.num_features, dropout_p = self.dropout_p, seq_length = self.seq_length, dtype = self.dtype).to(self.device)
        elif self.model_type == "transformer_no_decode":
            model = TransformerNoDecodeModel(num_features = self.dim_model, num_head = self.num_head, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, device = self.device, dtype = self.dtype, num_seqs = self.num_features).to(self.device)
        self.train(model)
        self.evaluate(model, self.checkpoint_folder)

if __name__ == "__main__":
    mainDir = "/media/nvme1/expansion/glycemic_health_data/physionet.org/files/big-ideas-glycemic-wearable/1.1.2/"
    obj = runModel(mainDir)
    obj.run()
