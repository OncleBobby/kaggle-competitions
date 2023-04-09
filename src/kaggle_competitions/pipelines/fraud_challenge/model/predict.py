import pandas as pd, logging, time, datetime
def train_model(x, y, model_name):
    from ..calibration import get_estimators
    model = get_estimators[model_name]
    model.fit(x, y[['fraud_flag']]) 
    return model
def predict(model, x):
    y_predict_proba =  model.predict_proba(x)
    df = pd.DataFrame(x['ID'].copy(), columns=['ID'])
    df['fraud_flag'] = [y[1] for y in y_predict_proba]
    df = df.reset_index()
    return df