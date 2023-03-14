import pandas, logging, datetime
# 2023-03-11 11:09:52,781 - root- INFO - 0.48523657061543934      Random Forest in 0:13:23.226058
# 2023-03-11 18:35:40,738 - root- INFO - 0.4972636360042977       Gradient Boosting in 7:25:47.956283
# 2023-03-11 18:38:06,242 - root- INFO - 0.471355317673354        Ada Boost in 0:02:25.500785
# 2023-03-11 20:54:13,962 - root- INFO - 0.4887293425180298       Extra Trees in 2:16:07.718478
# 2023-03-11 20:55:09,602 - root- INFO - 0.40575085612438866      Decision Tree in 0:00:55.639751
# 2023-03-11 20:55:27,772 - root- INFO - 0.5277355554836878       Hist Gradient Boosting in 0:00:18.168397
# 2023-03-11 20:57:36,318 - root- INFO - 0.41186680034065304      MLP in 0:02:08.545308

def calibrate_model(params, x_train, y_train, x_test, y_test):
    from sklearn.metrics import accuracy_score
    best_score = 0
    best_name = None
    lines = []
    target_field = params['target_field']
    id_field = params['id_field']
    for name in params['estimator_names']:
        start_time = datetime.datetime.now()
        estimator = _train(name, target_field, x_train, y_train)
        y_predict = _predict(estimator, id_field, target_field, x_test)
        score = accuracy_score(y_test[target_field].values.tolist(), y_predict[target_field].values.tolist())
        end_time = datetime.datetime.now()
        duration = (end_time - start_time)
        line = {
            'start_time': start_time, 
            'end_time': end_time, 
            'duration': str(duration), 
            'name': name, 
            'score': score
            }
        lines.append(line)
        logging.info(f'{score}\t{name} in {duration}')
        if score > best_score:
            best_score = score
            best_name = name
    logging.info(f'The winner is {best_name} with a score of {best_score} !')
    return pandas.DataFrame(lines)
def train(params, x, y):
    target_field = params['target_field']
    return _train(params['estimator_name'], target_field, x, y)
def _train(name, target_field, x, y):
    estimator = _get_estimators()[name]
    estimator.fit(x, y[target_field])
    return estimator
def predict(params, estimator, x):
    target_field = params['target_field']
    id_field = params['id_field']
    return _predict(estimator, id_field, target_field, x)
def _predict(estimator, id_field, target_field, x):
    y_predict = pandas.DataFrame(x[id_field].copy(), columns=[id_field])
    y_predict[target_field] = [y for y in estimator.predict(x)]
    y_predict = y_predict.reset_index()
    return y_predict
def _get_estimators():
    import sklearn.ensemble
    import sklearn.dummy
    import sklearn.tree
    import sklearn.calibration
    import sklearn.neural_network
    import sklearn.calibration
    return {
            'Dummy Classifier most_frequent': sklearn.dummy.DummyClassifier(strategy='most_frequent'),
            'Dummy Classifier prior': sklearn.dummy.DummyClassifier(strategy='prior'),
            'Dummy Classifier stratified': sklearn.dummy.DummyClassifier(strategy='stratified'),
            'Dummy Classifier uniform': sklearn.dummy.DummyClassifier(strategy='uniform'),
            'Dummy Classifier constant': sklearn.dummy.DummyClassifier(strategy='constant', constant=0),
            'Random Forest Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier()),
            'Gradient Boosting - log_loss': sklearn.ensemble.GradientBoostingClassifier(loss='log_loss'),
            'Gradient Boosting - deviance': sklearn.ensemble.GradientBoostingClassifier(loss='deviance'),
            'Gradient Boosting - exponential': sklearn.ensemble.GradientBoostingClassifier(loss='exponential'),
            'Gradient Boosting': sklearn.ensemble.GradientBoostingClassifier(),
            'Gradient Boosting Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.GradientBoostingClassifier()),
            'Ada Boost': sklearn.ensemble.AdaBoostClassifier(),
            'Ada Boost Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.AdaBoostClassifier()),
            'Bagging': sklearn.ensemble.BaggingClassifier(),
            'Bagging - Extra Trees': sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.ExtraTreesClassifier()),
            'Bagging - Random Forest': sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier()),
            'Bagging - Hist Gradient Boosting': sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.HistGradientBoostingClassifier()),
            'Bagging Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.BaggingClassifier()),
            'Extra Trees': sklearn.ensemble.ExtraTreesClassifier(),
            'Extra Trees Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier()),
            'Decision Tree': sklearn.tree.DecisionTreeClassifier(),
            'Decision Tree Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier()),
            'Hist Gradient Boosting': sklearn.ensemble.HistGradientBoostingClassifier(loss='log_loss'),
            'Hist Gradient Boosting - auto': sklearn.ensemble.HistGradientBoostingClassifier(loss='auto'),
            'Hist Gradient Boosting - categorical_crossentropy’': sklearn.ensemble.HistGradientBoostingClassifier(loss='categorical_crossentropy'),
            'Hist Gradient Boosting Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier()),
            'MLP': sklearn.neural_network.MLPClassifier(),
            'MLP Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier()),
            'Stacking 2': sklearn.ensemble.StackingClassifier([
                        ('gbc', sklearn.ensemble.GradientBoostingClassifier(loss='deviance')),
                        ('bc', sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier())),
                        ('etc', sklearn.ensemble.ExtraTreesClassifier())
                    ]),
            'Stacking 4.1': sklearn.ensemble.StackingClassifier([
                        ('gbc', sklearn.ensemble.GradientBoostingClassifier(loss='deviance')),
                        ('bc', sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier())),
                        ('etc', sklearn.ensemble.ExtraTreesClassifier()),
                        ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
                    ]),
            'Stacking 4': sklearn.ensemble.StackingClassifier([
                        ('gbc', sklearn.ensemble.GradientBoostingClassifier(loss='deviance')),
                        ('bc', sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier())),
                        ('etc', sklearn.ensemble.ExtraTreesClassifier()),
                        ('rfc', sklearn.ensemble.RandomForestClassifier()),
                        ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
                    ]),
            'Stacking': sklearn.ensemble.StackingClassifier(estimators = [
                        ('gbc', sklearn.ensemble.GradientBoostingClassifier(loss='deviance')),
                        ('abc', sklearn.ensemble.AdaBoostClassifier()),
                        ('bc', sklearn.ensemble.BaggingClassifier()),
                        ('etc', sklearn.ensemble.ExtraTreesClassifier()),
                        ('rfc', sklearn.ensemble.RandomForestClassifier()),
                        ('hgb', sklearn.ensemble.HistGradientBoostingClassifier()),
                        ('dtc', sklearn.tree.DecisionTreeClassifier()),
                        ('mlp', sklearn.neural_network.MLPClassifier())
                    ]),
            'Stacking Ref': sklearn.ensemble.StackingClassifier(estimators = [
                        ('gbc', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.GradientBoostingClassifier(loss='deviance'), cv=5, method='isotonic')),
                        ('abc', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.AdaBoostClassifier(), cv=5, method='isotonic')),
                        ('bc', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.BaggingClassifier(), cv=5, method='isotonic')),
                        ('etc', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier(), cv=5, method='isotonic')),
                        ('rfc', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
                        ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
                        ('dtc', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
                        ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
                    ])
                }
