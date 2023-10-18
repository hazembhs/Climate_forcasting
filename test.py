import torch
from data import dfFormat, trainX, valX, trainY_df, testX_real
from models import *
from climate_challenge_custom_metric import climate_metric_function
from utils import prediction_reshape
import pandas as pd
import csv
def model_score(model):
    trained = model(trainX).cpu().detach().numpy().reshape(-1, 16)
    means_train, variances_train = prediction_reshape(trained)
    pred_Y_train = pd.DataFrame({'MEAN':means_train, 'VARIANCE': variances_train})
    train_score = climate_metric_function(pred_Y_train, trainY_df)
    return train_score

def test_submit(model):
    pred = model(testX_real).cpu().detach().numpy().reshape(-1, 16)
    means, variances = prediction_reshape(pred)
    df = dfFormat
    df["MEAN"] = means
    df["VARIANCE"] = variances
    df.to_csv('pred.csv', index = True)


if __name__ == "__main__":
    model = LinearModel(254, 1).to("cuda")
    model.load_state_dict(torch.load('./model_state/model_1.pt'))
    s1 = model_score(model)
    print(f"Training score: {s1:.5f}")
    test_submit(model)