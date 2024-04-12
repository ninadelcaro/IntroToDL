import pickle
import os
import torch.nn as nn
import torch
import numpy as np


allaudios = [] # Creates an empty list
for root, dirs, files in os.walk("train"):
    i = 0
    for file in files:
        if file.endswith(".pkl"):
           audio = file
           openaudios = open(os.getcwd() + "/train/" + audio, 'rb')
           loadedaudios = pickle.load(openaudios)
           
           allaudios.append(loadedaudios)
           i += 1
           if i== 100:
               break

# split into X and Y, and train-test
audio_data = []
valence = []
for audio in allaudios:
    audio_data.append(audio['audio_data'])
    valence.append(audio['valence'])
    
size_train = int(round(len(allaudios) * 0.8))
X_train = np.array(audio_data[:size_train])
X_test = np.array(audio_data[size_train:])
y_train = np.array(valence[:size_train])
y_test = np.array(valence[size_train:])

# normalization
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x):
        with torch.no_grad():
            x = x - self.mean
            x = x / self.std
        return x
    
flatten = np.concatenate(X_train)
mean = np.mean(flatten)

std = flatten.std()
normalization = Normalization(mean, std)



