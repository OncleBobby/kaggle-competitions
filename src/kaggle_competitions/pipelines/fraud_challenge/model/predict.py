import numpy as np

def predict_benchmark(x):
    y = x[['ID']].copy(deep=True)
    y['fraud_flag'] = y.apply(lambda x: np.random.choice([0, 1]), axis=1)
    return y.reset_index(drop=True)
