import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
dfX = pd.read_csv("./train_X_origin.csv").set_index('ID')
dfY = pd.read_csv("./train_Y.csv").set_index("ID")
dfTest = pd.read_csv("./test_X.csv")
dfFormat = pd.read_csv("./format.csv").set_index("ID")
# print(dfFormat)
trainX_df = dfX.loc[dfX["DATASET"] < 4]
valX_df = dfX.loc[dfX["DATASET"] == 4]
trainY_df = dfY.loc[dfY["DATASET"] < 4]
valY_df = dfY.loc[dfY["DATASET"] == 4]
data = np.genfromtxt('./train_X.csv', delimiter=";")[1:, 2:]
data_real_test = np.genfromtxt('./test_X.csv', delimiter=";")[1:, 2:]
#print(np.shape(data))
#print(len(data[0]))
trainX_np = data[:12288, :254]
valX_np = data[12288:, :254]
trainY_np = data[:12288, -1]
valY_np = data[12288:, -1]
testX_real_np = data_real_test[:, :254]
#print(np.shape(trainX), np.shape(trainY))
#print(type(trainX[10, 10]))
#print(trainY[12])

if torch.cuda.is_available():
    trainX = torch.from_numpy(trainX_np).to("cuda").float()
    valX = torch.from_numpy(valX_np).to("cuda").float()
    valY = torch.from_numpy(valY_np).to("cuda").float().unsqueeze(1)
    trainY = torch.from_numpy(trainY_np).to("cuda").float().unsqueeze(1)
    testX_real = torch.from_numpy(testX_real_np).to("cuda").float()

train_data= TensorDataset(trainX, trainY)
val_data = TensorDataset(valX, valY)

train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
val_loader = DataLoader(val_data, batch_size = 64, shuffle = True)