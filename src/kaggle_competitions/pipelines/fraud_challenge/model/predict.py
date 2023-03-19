import pandas as pd, logging, time, datetime
def calibrate_model(x_train, y_train, x_test, y_test):
    import sklearn.ensemble
    import sklearn.tree
    import sklearn.calibration
    import sklearn.neural_network
    import sklearn.calibration
    from ..model.scope import score
    estimators = {
            'Gradient Boosting - log_loss': sklearn.ensemble.GradientBoostingClassifier(loss='log_loss'),
            # 'Gradient Boosting - deviance': sklearn.ensemble.GradientBoostingClassifier(loss='deviance'),
            # 'Gradient Boosting - exponential': sklearn.ensemble.GradientBoostingClassifier(loss='exponential'),
            # 'Gradient Boosting': sklearn.ensemble.GradientBoostingClassifier(),
            # 'Gradient Boosting Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.GradientBoostingClassifier()),
            # 'Ada Boost': sklearn.ensemble.AdaBoostClassifier(),
            # 'Ada Boost Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.AdaBoostClassifier()),
            # 'Bagging': sklearn.ensemble.BaggingClassifier(),
            # 'Bagging - Extra Trees': sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.ExtraTreesClassifier()),
            # 'Bagging - Random Forest': sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier()),
            # 'Bagging - Hist Gradient Boosting': sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.HistGradientBoostingClassifier()),
            # 'Bagging Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.BaggingClassifier()),
            # 'Extra Trees': sklearn.ensemble.ExtraTreesClassifier(),
            # 'Extra Trees Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier()),
            # 'Random Forest': sklearn.ensemble.RandomForestClassifier(),
            # 'Random Forest Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier()),
            # 'Decision Tree': sklearn.tree.DecisionTreeClassifier(),
            # 'Decision Tree Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier()),
            # 'Hist Gradient Boosting': sklearn.ensemble.HistGradientBoostingClassifier(),
            # 'Hist Gradient Boosting Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier()),
            # 'MLP': sklearn.neural_network.MLPClassifier(),
            # 'MLP Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier()),
            # 'Stacking 2': sklearn.ensemble.StackingClassifier([
            #             ('gb', sklearn.ensemble.GradientBoostingClassifier(loss='deviance')),
            #             ('bc', sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier())),
            #             ('etc', sklearn.ensemble.ExtraTreesClassifier())
            #         ]),
            # 'Stacking 4.1': sklearn.ensemble.StackingClassifier([
            #             ('gb', sklearn.ensemble.GradientBoostingClassifier(loss='deviance')),
            #             ('bc', sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier())),
            #             ('et', sklearn.ensemble.ExtraTreesClassifier()),
            #             ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
            #         ]),
            # 'Stacking 4': sklearn.ensemble.StackingClassifier([
            #             ('gb', sklearn.ensemble.GradientBoostingClassifier(loss='deviance')),
            #             ('b', sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier())),
            #             ('et', sklearn.ensemble.ExtraTreesClassifier()),
            #             ('rf', sklearn.ensemble.RandomForestClassifier()),
            #             ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
            #         ]),
            # 'Stacking': sklearn.ensemble.StackingClassifier(estimators = [
            #             ('gb', sklearn.ensemble.GradientBoostingClassifier(loss='deviance')),
            #             ('ab', sklearn.ensemble.AdaBoostClassifier()),
            #             ('b', sklearn.ensemble.BaggingClassifier()),
            #             ('et', sklearn.ensemble.ExtraTreesClassifier()),
            #             ('rf', sklearn.ensemble.RandomForestClassifier()),
            #             ('hgb', sklearn.ensemble.HistGradientBoostingClassifier()),
            #             ('dt', sklearn.tree.DecisionTreeClassifier()),
            #             ('mlp', sklearn.neural_network.MLPClassifier())
            #         ]),
            # 'Stacking Ref': sklearn.ensemble.StackingClassifier(estimators = [
            #             ('gb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.GradientBoostingClassifier(loss='deviance'), cv=5, method='isotonic')),
            #             ('ab', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.AdaBoostClassifier(), cv=5, method='isotonic')),
            #             ('b', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.BaggingClassifier(), cv=5, method='isotonic')),
            #             ('et', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier(), cv=5, method='isotonic')),
            #             ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
            #             ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
            #             ('dt', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
            #             ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
            #         ])
                }
    best_score = 0
    best_name = None
    for name, estimator in estimators.items():
        start_time = time.time()
        estimator.fit(x_train, y_train[['fraud_flag']])
        y_predict = pd.DataFrame(x_test['ID'].copy(), columns=['ID'])
        y_predict['fraud_flag'] = [y[1] for y in estimator.predict_proba(x_test)]
        y_predict = y_predict.reset_index()
        current_score = score(y_test, y_predict)
        logging.info(f'{current_score}\t{name} in {time.time() - start_time}s {datetime.datetime.fromtimestamp(time.time() - start_time).strftime("%M:%S.%f")}')
        if current_score > best_score:
            best_score = current_score
            best_name = name
    logging.info(f'The winner is {best_name} with a score of {best_score} !')
    return best_score
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

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    return CalibratedClassifierCV(estimator=RandomForestClassifier(), cv=5, method='isotonic')
    # return RandomForestClassifier()

    # from sklearn.ensemble import GradientBoostingClassifier
    # from sklearn.ensemble import AdaBoostClassifier
    # from sklearn.ensemble import BaggingClassifier
    # from sklearn.ensemble import ExtraTreesClassifier
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.ensemble import HistGradientBoostingClassifier
    # from sklearn.neural_network import MLPClassifier
    # from sklearn.ensemble import StackingClassifier as Model # 0.21714637918918994
    # # CalibratedClassifierCV 0.32726128502644053
    # estimators = [
    #         ('gb', CalibratedClassifierCV(GradientBoostingClassifier(loss='deviance'), cv=5, method='isotonic')),
    #         ('ab', CalibratedClassifierCV(AdaBoostClassifier(), cv=5, method='isotonic')),
    #         ('b', CalibratedClassifierCV(BaggingClassifier(), cv=5, method='isotonic')),
    #         ('et', CalibratedClassifierCV(ExtraTreesClassifier(), cv=5, method='isotonic')),
    #         ('rf', CalibratedClassifierCV(RandomForestClassifier(), cv=5, method='isotonic')),
    #         ('dt', CalibratedClassifierCV(DecisionTreeClassifier(), cv=5, method='isotonic')),
    #         ('hgb', CalibratedClassifierCV(HistGradientBoostingClassifier(), cv=5, method='isotonic')),
    #         ('mlp', CalibratedClassifierCV(MLPClassifier(), cv=5, method='isotonic'))
    #     ]
    # return Model(estimators=estimators)



