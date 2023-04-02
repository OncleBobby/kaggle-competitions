import pandas, logging, datetime, numpy
from sklearn.metrics import average_precision_score

def calibrate_model(params, x_train, y_train, x_test, y_test):
    best_score = 0
    best_model = None
    best_name = None
    lines = []
    for name in params['estimator_names']:
        start_time = datetime.datetime.now()
        estimator = _get_estimators()[name]
        estimator.fit(x_train, y_train)
        y_predict = estimator.predict_proba(x_test)
        current_score = score(y_test, y_predict)
        end_time = datetime.datetime.now()
        duration = (end_time - start_time)
        line = {
            'start_time': start_time, 
            'end_time': end_time, 
            'duration': str(duration), 
            'name': name, 
            'score': current_score
            }
        lines.append(line)
        logging.info(f'{current_score}\t{name} in {duration}')
        if current_score > best_score:
            best_score = current_score
            best_name = name
            best_model = estimator
    logging.info(f'The winner is {best_name} with a score of {best_score} !')
    return best_model, pandas.DataFrame(lines)
def score(y_true, y_pred_proba):
    ''' 
    Return the area under the Precision-Recall curve.  
    Args:
        - y_true (pd.DataFrame): Dataframe with a unique identifier for each observation (first column) and the ground truth observations (second column).
        - y_pred_proba (pd.DataFrame): Dataframe with a unique identifier for each observation (first column) and the predicted probabilities estimates for the minority class (second column).
    Returns:
        float
    '''   
    y_true_sorted = y_true.sort_values(by='ID').reset_index(drop=True)[['ID', 'fraud_flag']]
    y_pred_proba_sorted = y_pred_proba.sort_values(by='ID').reset_index(drop=True)[['ID', 'fraud_flag']]

    score = average_precision_score(numpy.ravel(y_true_sorted.iloc[:, 1]), numpy.ravel(y_pred_proba_sorted.iloc[:, 1]))
    return score
def train(params, x, y):
    estimator = _get_estimators()[params['estimator_name']]
    estimator.fit(x, y[['fraud_flag']])
    return estimator
def predict(params, estimator, x):
    target_field = params['target_field']
    id_field = params['id_field']
    return _predict(estimator, id_field, target_field, x)
def _predict(estimator, id_field, target_field, x):
    y_predict = pandas.DataFrame(x[id_field].copy(), columns=[id_field])
    y_predict[target_field] = [y for y in estimator.predict(x)]
    y_predict = y_predict.reset_index(drop=True)
    return y_predict
def _get_estimators():
    import sklearn.ensemble, sklearn.dummy, sklearn.tree, sklearn.calibration, sklearn.neural_network, sklearn.calibration, sklearn.gaussian_process
    import sklearn.linear_model, sklearn.multiclass, sklearn.svm, sklearn.naive_bayes, sklearn.neighbors, sklearn.semi_supervised
    from .model.sklearn_estimator import SklearnEstimator
    from .model.keras_estimator import KerasEstimator
    estimators = {
            'Dummy Classifier most_frequent': sklearn.dummy.DummyClassifier(strategy='most_frequent'),
            'Dummy Classifier prior': sklearn.dummy.DummyClassifier(strategy='prior'),
            'Dummy Classifier stratified': sklearn.dummy.DummyClassifier(strategy='stratified'),
            'Dummy Classifier uniform': sklearn.dummy.DummyClassifier(strategy='uniform'),
            'Dummy Classifier constant': sklearn.dummy.DummyClassifier(strategy='constant', constant=0),
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
            'Hist Gradient Boosting - categorical_crossentropy': sklearn.ensemble.HistGradientBoostingClassifier(loss='categorical_crossentropy'),
            'Hist Gradient Boosting Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier()),
            'MLP': sklearn.neural_network.MLPClassifier(),
            'MLP Calibrated': sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier()),
            'Complement NB': sklearn.naive_bayes.ComplementNB(),
            'Gaussian NB': sklearn.naive_bayes.GaussianNB(),
            'KNeighbors': sklearn.neighbors.KNeighborsClassifier(),
            'Self Training': sklearn.semi_supervised.SelfTrainingClassifier(sklearn.svm.SVC(probability=True, gamma="auto")),
            'Stacking_rf_et_hgb': sklearn.ensemble.StackingClassifier([
                        ('rf', sklearn.ensemble.RandomForestClassifier()),
                        ('et', sklearn.ensemble.ExtraTreesClassifier()),
                        ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
                    ]),
            'Stacking_rfc_etc_hgbc': sklearn.ensemble.StackingClassifier([
                        ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier())),
                        ('et', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier())),
                        ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier()))
                    ]),
            'Stacking_ab_b_hgb': sklearn.ensemble.StackingClassifier([
                        ('ab', sklearn.ensemble.AdaBoostClassifier()),
                        ('bc', sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier())),
                        ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
                    ]),
            'Stacking_gb_ab_b_hgb': sklearn.ensemble.StackingClassifier([
                        ('gb', sklearn.ensemble.GradientBoostingClassifier(loss='deviance')),
                        ('ab', sklearn.ensemble.AdaBoostClassifier()),
                        ('bc', sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier())),
                        ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
                    ]),
            'Stacking_gb_ab_b_hgb': sklearn.ensemble.StackingClassifier([
                        ('ab', sklearn.ensemble.AdaBoostClassifier()),
                        ('bc', sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier())),
                        ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
                    ]),
            'Stacking_AB_GB_HGB': sklearn.ensemble.StackingClassifier([
                        ('ab', sklearn.ensemble.AdaBoostClassifier()),
                        ('gb', sklearn.ensemble.GradientBoostingClassifier(loss='log_loss')),
                        ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier()))
                    ]),
            'Stacking_ab_gb_hgb': sklearn.ensemble.StackingClassifier([
                        ('ab', sklearn.ensemble.AdaBoostClassifier()),
                        ('gb', sklearn.ensemble.GradientBoostingClassifier(loss='log_loss')),
                        ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
                    ]),
            'Stacking 4.1': sklearn.ensemble.StackingClassifier([
                        ('gb', sklearn.ensemble.GradientBoostingClassifier(loss='deviance')),
                        ('b', sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier())),
                        ('et', sklearn.ensemble.ExtraTreesClassifier()),
                        ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
                    ]),
            'Stacking 4': sklearn.ensemble.StackingClassifier([
                        ('gb', sklearn.ensemble.GradientBoostingClassifier(loss='deviance')),
                        ('b', sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier())),
                        ('et', sklearn.ensemble.ExtraTreesClassifier()),
                        ('rf', sklearn.ensemble.RandomForestClassifier()),
                        ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
                    ]),
            'Stacking': sklearn.ensemble.StackingClassifier(estimators = [
                        ('gb', sklearn.ensemble.GradientBoostingClassifier(loss='deviance')),
                        ('ab', sklearn.ensemble.AdaBoostClassifier()),
                        ('b', sklearn.ensemble.BaggingClassifier()),
                        ('et', sklearn.ensemble.ExtraTreesClassifier()),
                        ('rf', sklearn.ensemble.RandomForestClassifier()),
                        ('hgb', sklearn.ensemble.HistGradientBoostingClassifier()),
                        ('dt', sklearn.tree.DecisionTreeClassifier()),
                        ('mlp', sklearn.neural_network.MLPClassifier())
                    ]),
            'Stacking Ref': sklearn.ensemble.StackingClassifier(estimators = [
                        ('ab', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.AdaBoostClassifier(), cv=5, method='isotonic')),
                        ('b', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.BaggingClassifier(), cv=5, method='isotonic')),
                        ('et', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier(), cv=5, method='isotonic')),
                        ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
                        ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
                        ('dt', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
                        ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
                    ]),
            'Stacking_gb_b_et_rf_hgb': sklearn.ensemble.StackingClassifier(estimators = [
                        ('gb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.GradientBoostingClassifier(loss='deviance'), cv=5, method='isotonic')),
                        ('b', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.BaggingClassifier(), cv=5, method='isotonic')),
                        ('et', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier(), cv=5, method='isotonic')),
                        ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
                        ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic'))
                    ]),
            'Stacking_ab_b_et_rf_hgb_dt_mlp': sklearn.ensemble.StackingClassifier(estimators = [
                        ('ab', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.AdaBoostClassifier(), cv=5, method='isotonic')),
                        ('b', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.BaggingClassifier(), cv=5, method='isotonic')),
                        ('et', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier(), cv=5, method='isotonic')),
                        ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
                        ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
                        ('dt', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
                        ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
                    ]),
            'Stacking_b_et_rf_hgb_dt_mlp': sklearn.ensemble.StackingClassifier(estimators = [
                        ('b', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.BaggingClassifier(), cv=5, method='isotonic')),
                        ('et', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier(), cv=5, method='isotonic')),
                        ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
                        ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
                        ('dt', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
                        ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
                    ]),
            'Stacking_et_rf_hgb_dt_mlp': sklearn.ensemble.StackingClassifier(estimators = [
                        ('et', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier(), cv=5, method='isotonic')),
                        ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
                        ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
                        ('dt', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
                        ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
                    ]),
            'Stacking_rf_hgb_dt_mlp': sklearn.ensemble.StackingClassifier(estimators = [
                        ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
                        ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
                        ('dt', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
                        ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
                    ]),
            'Stacking_hgb_dt_mlp': sklearn.ensemble.StackingClassifier(estimators = [
                        ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
                        ('dt', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
                        ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
                    ])                      
        }
    for k in estimators.keys():
        estimators[k] = SklearnEstimator(estimators[k])
    estimators['relu'] = KerasEstimator('relu', 100, 100)
    estimators['sigmoid'] = KerasEstimator('sigmoid', 100, 100)
    estimators['softmax'] = KerasEstimator('softmax', 100, 100)
    estimators['softplus'] = KerasEstimator('softplus', 100, 100)
    estimators['softsign'] = KerasEstimator('softsign', 100, 100)
    estimators['tanh'] = KerasEstimator('tanh', 100, 100)
    estimators['selu'] = KerasEstimator('selu', 100, 100)
    estimators['relu'] = KerasEstimator('relu', 100, 100)
    estimators['softplus'] = KerasEstimator('softplus', 100, 100)
    estimators['elu'] = KerasEstimator('elu', 100, 100)
    return estimators
