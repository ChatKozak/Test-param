###-----------------
### Import Libraries
###-----------------

import os
#import numpy as np
import pandas as pd

from collections.abc import Callable
from typing import Union

value: Union[int, str] = 1



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

###----------------------
### Some basic parameters
###----------------------

inpDir = '..'
outDir = '..'

RANDOM_STATE = 177013
torch.manual_seed(RANDOM_STATE) # Set Random Seed for reproducible  results

EPOCHS = 50 # number of epochs
ALPHA = 0.001 # learning rate
TEST_SIZE = 0.2

print ('Is CUDA available: ', torch.cuda.is_available())

print ('CUDA version: ', torch.version.cuda )

# print ('Current Device ID: ', torch.cuda.current_device())

# print ('Name of the CUDA device: ', torch.cuda.get_device_name(torch.cuda.current_device()))

# Get cpu or gpu device for training.

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")


data_df = pd.read_csv("fifa_2019.csv")
data_df.shape

# removing rows with position == null
data_df = data_df[data_df["Position"].notnull()]
data_df.head()


# Following columns appear to be relevant for our analysis
rel_cols = ["Position", 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
            'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
            'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
            'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
            'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
            'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
            'GKKicking', 'GKPositioning', 'GKReflexes']


goalkeeper = 'GK'
forward = ['ST', 'LW', 'RW', 'LF', 'RF', 'RS','LS', 'CF']
midfielder = ['CM','RCM','LCM', 'CDM','RDM','LDM', 'CAM', 'LAM', 'RAM', 'RM', 'LM']
defender = ['CB', 'RCB', 'LCB', 'LWB', 'RWB', 'LB', 'RB']

#Assign labels to goalkeepers
data_df.loc[data_df["Position"] == "GK", "Position"] = 0

#Defenders
data_df.loc[data_df["Position"].isin(defender), "Position"] = 1

#Midfielders
data_df.loc[data_df["Position"].isin(midfielder), "Position"] = 2

#Forward
data_df.loc[data_df["Position"].isin(forward), "Position"] = 3

# Helps in preventing pandas from complaining while get_dummies
data_df['Position'] = pd.to_numeric(data_df['Position'], downcast="integer")





# Keeping relevent columns.
data_df = data_df[rel_cols]
data_df.head()


feature = data_df.drop('Position', axis = 1).to_numpy()
label = data_df['Position'].to_numpy()



X_train, X_test, y_train, y_test = train_test_split(feature, label,
                                                    stratify=label,
                                                    test_size=TEST_SIZE, 
                                                    random_state=RANDOM_STATE )

X_train.shape, X_test.shape, y_train.shape, y_test.shape



features = torch.tensor(X_train, dtype=torch.float32, device=device)
labels = torch.tensor(y_train, dtype=torch.int64, device=device)
features


# nn.Sequential network with nn.Linear layers


net = nn.Sequential(nn.Linear(X_train.shape[1],16),nn.ReLU(),nn.Linear(16,4),nn.Softmax(dim=1))
net=net.to(device)



net



# Test its working
net(features).shape



#define the loss fn and optimiser
loss_fn=nn.CrossEntropyLoss() # cross entopy loss
optimiser=torch.optim.Adam(net.parameters(),lr=ALPHA) #optimiser


for e in range(EPOCHS):
    #Zero your Gradients for Every batch!
    optimiser.zero_grad()
    
    #Make prediction for this batch
    outputs=net(features)
    
    #compute the loss and its gradients
    train_loss=loss_fn(outputs,labels)
    train_loss.backward()
    
    #adjust learning rates
    optimiser.step()
    
    print(train_loss.item())


test_features=torch.tensor(X_test,device=device,dtype=torch.float32)
test_labels=torch.tensor(y_test,device=device,dtype=torch.int64)

output=net(test_features)
y_pred=output.argmax(dim=1).cpu().numpy()

print(accuracy_score(y_test,y_pred))
