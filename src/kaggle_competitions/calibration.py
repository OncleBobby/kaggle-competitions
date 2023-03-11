import pandas, logging, time, datetime
    # from sklearn.ensemble import RandomForestClassifier as Model # 0.24190277816810518
    # from sklearn.ensemble import StackingClassifier as Model
    # from sklearn.neural_network import MLPClassifier as Model # 0.04475057636819481
    # from sklearn.neighbors import KNeighborsClassifier as Model # 0.07055742578682545
    # from sklearn.neighbors import RadiusNeighborsClassifier as Model
    # from sklearn.calibration import  as Model # 0.01688744670335069
    # from sklearn.gaussian_process import GaussianProcessClassifier as Model
    # model = sklearn.ensemble.RandomForestClassifier()

def calibrate_model(estimators, target_field, id_field, x_train, y_train, x_test, y_test):
    from sklearn.metrics import accuracy_score
    best_score = 0
    best_name = None
    lines = []
    for name, estimator in estimators.items():
        start_time = time.time()
        estimator.fit(x_train, y_train[target_field])
        y_predict = pandas.DataFrame(x_test[id_field].copy(), columns=[id_field])
        y_predict[target_field] = [y for y in estimator.predict(x_test)]
        y_predict = y_predict.reset_index()
        score = accuracy_score(y_test[target_field].values.tolist(), y_predict[target_field].values.tolist())

        end_time = time.time()
        duration = (time.time() - start_time)
        duration_str = datetime.datetime.fromtimestamp(duration).strftime('%M:%S.%f')
        line = {'start_time': start_time, 'end_time': end_time, 'duration': duration, 'duration_str': duration_str, 'name': name, 'score': score}
        lines.append(line)
        logging.info(f'{score}\t{name} in {time.time() - start_time}s {duration_str}')
        if score > best_score:
            best_score = score
            best_name = name
    logging.info(f'The winner is {best_name} with a score of {best_score} !')
    return pandas.DataFrame(lines)
def get_estimators(estimator_names):
    import sklearn.ensemble
    import sklearn.tree
    import sklearn.calibration
    import sklearn.neural_network
    import sklearn.calibration
    estimators = {
            'Random Forest': sklearn.ensemble.RandomForestClassifier(),
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
    return { name: estimators[name] for name in estimator_names}
