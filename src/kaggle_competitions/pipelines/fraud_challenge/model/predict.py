import pandas as pd

def train_model(x, y):
    # from sklearn.ensemble import GradientBoostingClassifier as Model # 0.14256094886399923
    # from sklearn.ensemble import AdaBoostClassifier as Model # 0.07457939372227873
    # from sklearn.ensemble import BaggingClassifier as Model # 0.1938640497809566
    # from sklearn.ensemble import ExtraTreesClassifier as Model # 0.20789089489415438
    from sklearn.ensemble import RandomForestClassifier as Model # 0.24190277816810518
    # from sklearn.ensemble import StackingClassifier as Model
    # from sklearn.ensemble import VotingClassifier as Model
    # from sklearn.ensemble import HistGradientBoostingClassifier as Model # 0.2377164908158746
    
    
    model = Model()
    y = y[['fraud_flag']]
    model.fit(x, y) 
    return model
def predict(model, x):
    y_predict_proba =  model.predict_proba(x)
#    y_predict =  model.predict(x)
    df = pd.DataFrame(x['ID'].copy(), columns=['ID'])
    df['fraud_flag'] = [y[1] for y in y_predict_proba]

    df = df.reset_index()

    return df