import numpy as np, pandas as pd

def train_decision_tree_regressor(x, y):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=1)
    model.fit(x, y) 
    return model
def predict(model, x):
    y_predict =  pd.DataFrame(model.predict(x), columns=['ID', 'fraud_flag']).astype({'ID':'int', 'fraud_flag': 'int'})
    print(f"y_predict={y_predict.tail}")
    return y_predict