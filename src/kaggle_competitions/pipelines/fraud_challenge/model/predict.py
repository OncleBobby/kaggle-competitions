import pandas as pd

def train_decision_tree_regressor(x, y):
    from sklearn.ensemble import GradientBoostingClassifier as Model
    model = Model()
    y = y[['fraud_flag']]
    model.fit(x, y) 
    return model
def predict(model, x):
    y_predict_proba =  model.predict_proba(x)
#    y_predict =  model.predict(x)
    df = pd.DataFrame(x['ID'].copy(), columns=['ID'])
    df['fraud_flag'] = [y[1] for y in y_predict_proba]
    return df