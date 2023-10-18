import torch
import torch.nn as nn
import torch.optim as op
from models import *
from data import train_loader, val_loader
from train import train_loop
model_1 = LinearModel(254, 1).to("cuda")
optimizer = op.Adam(model_1.parameters(), lr = 0.001)
loss_fn = nn.MSELoss()

if __name__ == "__main__":
    train_loop(
        model = model_1,
        optimizer=optimizer,
        loss_fn = loss_fn,
        epochs = 10,
        train_loader = train_loader,
        val_loader = val_loader
    )

    torch.save(model_1.state_dict(), './model_state/model_1.pt')
# print(list(model_1.parameters()))

