import pandas as pd
import numpy as np

from climate_challenge_custom_metric import climate_metric_function

dfX = pd.read_csv("./train_X_origin.csv").set_index('ID')
dfY = pd.read_csv("./train_Y.csv").set_index("ID")
test_X = dfX.loc[dfX["DATASET"] == 4]
test_Y = dfY.loc[dfY["DATASET"] == 4]
train_X = dfX.loc[dfX["DATASET"]<4]
train_Y = dfX.loc[dfX["DATASET"]<4]
test_X_history = test_X[test_X.TIME<10]
test_X_predictions = test_X[test_X.TIME==10]
train_X_predictions = train_X[train_X.TIME==10]
def aggregate_healpix(data):
    dfs = []
    for i in data.DATASET.unique():
        df = data[data.DATASET == i]
        means = np.mean(df.VALUE.to_numpy().reshape(192,16),1)
        vars = np.var(df.VALUE.to_numpy().reshape(192,16),1)
        df = pd.DataFrame()
        df['MEAN'] = means
        df['VARIANCE'] = vars
        df['DATASET'] = i
        df['POSITION'] = [i for i in range(len(df))]
        df = df[['DATASET', 'POSITION', 'MEAN', 'VARIANCE']]
        dfs.append(df)
    return pd.concat(dfs)

def benchmark_1(historical_data):
    preds_3072 = historical_data.groupby(['DATASET', 'POSITION']).VALUE.last().reset_index()
    return aggregate_healpix(preds_3072)

def benchmark_2(predictions_data):
    y_pred = predictions_data.groupby(['DATASET', 'POSITION']).VALUE.mean().reset_index()
    y_pred = y_pred.rename(columns = {'VALUE' : 'MEAN'})
    y_pred['VARIANCE'] = predictions_data.groupby(['DATASET', 'POSITION']).VALUE.var().reset_index().VALUE
    return y_pred

def pivot_predictions_data(predictions_data):
    df = predictions_data[predictions_data.TIME == 10].drop('TIME', axis = 1)
    return df.pivot(index = ["DATASET", "POSITION"], columns=["MODEL"], values=["VALUE"]).reset_index()


print(climate_metric_function(benchmark_1(test_X_history), test_Y))
print(climate_metric_function(benchmark_2(test_X_predictions), test_Y))
print(climate_metric_function(benchmark_2(train_X_predictions), train_Y))
