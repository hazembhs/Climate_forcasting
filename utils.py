def prediction_reshape(data):
    means = [row.mean() for row in data]

# Calculate squared errors
    squared_errors = [(row-mean)**2 for row, mean in zip(data, means)]

# Calculate "mean for each row of squared errors" (aka the variance)
    variances = [row.mean() for row in squared_errors]

    return means, variances


def index_til_exceed(aa, lim):
    sum = 0
    for k in range(len(aa)):
        sum+=aa[k]
        if sum > lim:
            return k
    return len(aa)-1

