import pandas as pd, logging

def train_model(x, y):
    model = _get_model()
    model.fit(x, y[['fraud_flag']]) 
    return model
def predict(model, x):
    y_predict_proba =  model.predict_proba(x)
#    y_predict =  model.predict(x)
    df = pd.DataFrame(x['ID'].copy(), columns=['ID'])
    df['fraud_flag'] = [y[1] for y in y_predict_proba]
    df = df.reset_index()
    return df
def _get_model():
    import sklearn.ensemble
    # from sklearn.ensemble import GradientBoostingClassifier as Model # 0.14256094886399923
    # from sklearn.ensemble import AdaBoostClassifier as Model # 0.07457939372227873
    # from sklearn.ensemble import BaggingClassifier as Model # 0.1938640497809566
    # from sklearn.ensemble import ExtraTreesClassifier as Model # 0.20789089489415438
    # from sklearn.ensemble import RandomForestClassifier as Model # 0.24190277816810518
    # from sklearn.ensemble import StackingClassifier as Model
    # from sklearn.ensemble import HistGradientBoostingClassifier as Model # 0.2377164908158746
    # from sklearn.tree import DecisionTreeClassifier as Model # 0.0755061564665697
    # from sklearn.tree import ExtraTreeClassifier as Model # 0.08096637219027183
    # from sklearn.neural_network import MLPClassifier as Model # 0.04475057636819481
    # from sklearn.neighbors import KNeighborsClassifier as Model # 0.07055742578682545
    # from sklearn.neighbors import RadiusNeighborsClassifier as Model
    # from sklearn.calibration import CalibratedClassifierCV as Model # 0.01688744670335069
    # from sklearn.gaussian_process import GaussianProcessClassifier as Model
    # model = sklearn.ensemble.RandomForestClassifier()
    # return model
    # return Model()

    # from sklearn.multiclass import OneVsRestClassifier as Model # 0.23834318037776012 RandomForestClassifier
    # from sklearn.ensemble import RandomForestClassifier
    # return Model(estimator=RandomForestClassifier())
    # return Model(base_estimator=RandomForestClassifier())

    # from sklearn.ensemble import BaggingClassifier as Model # 0.22919308219532086
    # from sklearn.ensemble import RandomForestClassifier
    # return Model(estimator=RandomForestClassifier())

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import StackingClassifier as Model # 0.21714637918918994
    estimators = [
            ('gbc', GradientBoostingClassifier()),
            ('abc', AdaBoostClassifier()),
            ('bc', BaggingClassifier()),
            ('etc', ExtraTreesClassifier()),
            ('rfc', RandomForestClassifier()),
            ('dtc', DecisionTreeClassifier()),
            ('hgb', HistGradientBoostingClassifier()),
            ('mlp', MLPClassifier())
        ]
    return Model(estimators=estimators)



