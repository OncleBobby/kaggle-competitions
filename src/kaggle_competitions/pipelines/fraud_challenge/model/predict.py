import pandas as pd, logging, time, datetime

def train_model(x, y, parameters):
    model_name = parameters['estimator_name']
    from ..calibration import get_estimators
    model = get_estimators()[model_name]
    model.fit(x, y[['fraud_flag']]) 
    return model
def predict(model, x):
    return model.predict_proba(x)