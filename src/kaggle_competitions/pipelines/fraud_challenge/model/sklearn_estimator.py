
import pandas, logging
from .estimator import Estimator
class SklearnEstimator(Estimator):
    def __init__(self, estimator):
        self.estimator = estimator
    def fit(self, x, y):
        self.estimator.fit(x, y[['fraud_flag']])
    def predict(self, x):
        return self.estimator.predict(x)
    def predict_proba(self, x):
        y_predict = pandas.DataFrame(x['ID'].copy(), columns=['ID'])
        y_predict['fraud_flag'] = [y[1] for y in self.estimator.predict_proba(x)]
        y_predict = y_predict.reset_index()
        return y_predict