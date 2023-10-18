from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from data import trainX_np, trainY_np
from utils import prediction_reshape
for k in range(len(trainX_np[0,:])):
    x, _ = prediction_reshape(trainX_np[:, k].reshape(-1, 16))
    y, _ = prediction_reshape(trainY_np[:].reshape(-1, 16))
    plt.scatter(x, y, alpha = 0.1)
    plt.savefig(f'./plots/var{k}.png')
    plt.clf()
    print(f"var{k} plotted")