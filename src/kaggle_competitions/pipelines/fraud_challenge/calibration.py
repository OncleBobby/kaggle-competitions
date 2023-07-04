import pandas, logging, datetime, numpy, itertools
from sklearn.metrics import average_precision_score

def calibrate_model(params, x_train, y_train, x_test, y_test):
    best_score = 0
    best_model = None
    best_name = None
    lines = []
    i = 0
    estimators = get_estimators()
    nbr = len(estimators.keys())
    # for name in params['estimator_names']:
    for name, estimator in estimators.items():
        i = i + 1
        start_time = datetime.datetime.now()
        # estimators = get_estimators()
        # if not name in estimators:
        #     logging.warning(f'{name} is not define.')
        #     continue
        # estimator = estimators[name]
        estimator.fit(x_train.fillna(0), y_train)
        y_predict = estimator.predict_proba(x_test.fillna(0))
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
        logging.info(f'{i}/{nbr} - {current_score}\t{name} in {duration}')
        if current_score > best_score:
            best_score = current_score
            best_name = name
            best_model = estimator
    logging.info(f'The winner is {best_name} with a score of {best_score} !')
    return best_model, best_name, pandas.DataFrame(lines)
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
    estimator = get_estimators()[params['estimator_name']]
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
def get_estimators():
    estimators = get_sklearn_estimators()
    # estimators.update(get_keras_estimators())
    return estimators
def get_sklearn_estimators():
    import sklearn.ensemble, sklearn.dummy, sklearn.tree, sklearn.calibration, sklearn.neural_network, sklearn.calibration, sklearn.gaussian_process
    import sklearn.linear_model, sklearn.multiclass, sklearn.svm, sklearn.naive_bayes, sklearn.neighbors, sklearn.semi_supervised
    from .model.sklearn_estimator import SklearnEstimator
    import xgboost
    
    basic_estimators = {
            # 'Dummy most frequent': sklearn.dummy.DummyClassifier(strategy='most_frequent'),
            # 'Dummy prior': sklearn.dummy.DummyClassifier(strategy='prior'),
            # 'Dummy stratified': sklearn.dummy.DummyClassifier(strategy='stratified'),
            # 'Dummy uniform': sklearn.dummy.DummyClassifier(strategy='uniform'),
            # 'Dummy constant': sklearn.dummy.DummyClassifier(strategy='constant', constant=0),
            # 'XGB': xgboost.XGBClassifier(),
            'Random Forest': sklearn.ensemble.RandomForestClassifier(),
            # 'Gradient Boosting log_loss': sklearn.ensemble.GradientBoostingClassifier(loss='log_loss'),
            # 'Gradient Boosting deviance': sklearn.ensemble.GradientBoostingClassifier(loss='deviance'),
            # 'Gradient Boosting exponential': sklearn.ensemble.GradientBoostingClassifier(loss='exponential'),
            'Gradient Boosting': sklearn.ensemble.GradientBoostingClassifier(),
            'Ada Boost': sklearn.ensemble.AdaBoostClassifier(),
            'Bagging': sklearn.ensemble.BaggingClassifier(),
            # 'Bagging Extra Trees': sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.ExtraTreesClassifier()),
            # 'Bagging Random Forest': sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier()),
            # 'Bagging Hist Gradient Boosting': sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.HistGradientBoostingClassifier()),
            'Extra Trees': sklearn.ensemble.ExtraTreesClassifier(),
            'Decision Tree': sklearn.tree.DecisionTreeClassifier(),
            # 'Hist Gradient Boosting': sklearn.ensemble.HistGradientBoostingClassifier(loss='log_loss'),
            # 'Hist Gradient Boosting auto': sklearn.ensemble.HistGradientBoostingClassifier(loss='auto'),
            # 'Hist Gradient Boosting categorical_crossentropy': sklearn.ensemble.HistGradientBoostingClassifier(loss='categorical_crossentropy'),
            # 'MLP': sklearn.neural_network.MLPClassifier(),
            'KNeighbors': sklearn.neighbors.KNeighborsClassifier()           
        }
    estimators = {}
    basic_estimators = {convert_to_short_name(name) : estimator for name, estimator in basic_estimators.items()}
    estimators = {convert_to_short_name(name) : estimator for name, estimator in basic_estimators.items()}
    calibrated_estimators = {}
    # for name, estimator in basic_estimators.items():
    #     calibrated_estimators[f"{name}C"] = sklearn.calibration.CalibratedClassifierCV(estimator)
    
    stacking_estimators = {}
    for names in itertools.permutations([e for e in list(basic_estimators.keys())], 2):
        names = list(names)
        names.sort()
        estimators_for_stacking = []
        estimators_for_stacking.append(('x', xgboost.XGBClassifier()))
        estimators_for_stacking.append(('hgb', sklearn.ensemble.HistGradientBoostingClassifier(loss='log_loss')))
        for name in names:
            estimators_for_stacking.append((name, basic_estimators[name]))
        names.append('x')
        names.append('hgb')
        join_name = f"{'_'.join(names)}"
        if not join_name in stacking_estimators:
            stacking_estimators[join_name] = sklearn.ensemble.StackingClassifier(estimators_for_stacking)

        stacking_estimators['Ref_ab_b_et_rf_hgb_dt_mlp'] = sklearn.ensemble.StackingClassifier(estimators = [
                        ('ab', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.AdaBoostClassifier(), cv=5, method='isotonic')),
                        ('b', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.BaggingClassifier(), cv=5, method='isotonic')),
                        ('et', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier(), cv=5, method='isotonic')),
                        ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
                        ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
                        ('dt', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
                        ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
                    ])
        stacking_estimators['ab_b_et_rf_hgb_dt_mlp_x'] = sklearn.ensemble.StackingClassifier(estimators = [
                        ('ab', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.AdaBoostClassifier(), cv=5, method='isotonic')),
                        ('b', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.BaggingClassifier(), cv=5, method='isotonic')),
                        ('et', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier(), cv=5, method='isotonic')),
                        ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
                        ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
                        ('dt', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
                        ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic')),
                        ('x',  xgboost.XGBClassifier())
                    ])


    # stacking_estimators = {
    #         'rf_et_hgb': sklearn.ensemble.StackingClassifier([
    #                     ('rf', sklearn.ensemble.RandomForestClassifier()),
    #                     ('et', sklearn.ensemble.ExtraTreesClassifier()),
    #                     ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
    #                 ]),
    #         'rfc_etc_hgbc': sklearn.ensemble.StackingClassifier([
    #                     ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier())),
    #                     ('et', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier())),
    #                     ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier()))
    #                 ]),
    #         'ab_b_hgb': sklearn.ensemble.StackingClassifier([
    #                     ('ab', sklearn.ensemble.AdaBoostClassifier()),
    #                     ('bc', sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier())),
    #                     ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
    #                 ]),
    #         'gb_ab_b_hgb': sklearn.ensemble.StackingClassifier([
    #                     ('gb', sklearn.ensemble.GradientBoostingClassifier(loss='deviance')),
    #                     ('ab', sklearn.ensemble.AdaBoostClassifier()),
    #                     ('bc', sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier())),
    #                     ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
    #                 ]),
    #         'gb_ab_b_hgb': sklearn.ensemble.StackingClassifier([
    #                     ('ab', sklearn.ensemble.AdaBoostClassifier()),
    #                     ('bc', sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier())),
    #                     ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
    #                 ]),
    #         'AB_GB_HGB': sklearn.ensemble.StackingClassifier([
    #                     ('ab', sklearn.ensemble.AdaBoostClassifier()),
    #                     ('gb', sklearn.ensemble.GradientBoostingClassifier(loss='log_loss')),
    #                     ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier()))
    #                 ]),
    #         'ab_gb_hgb': sklearn.ensemble.StackingClassifier([
    #                     ('ab', sklearn.ensemble.AdaBoostClassifier()),
    #                     ('gb', sklearn.ensemble.GradientBoostingClassifier(loss='log_loss')),
    #                     ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
    #                 ]),
    #         'gb_ab_b_et_rf_hgb_dt_mlp': sklearn.ensemble.StackingClassifier([
    #                     ('gb', sklearn.ensemble.GradientBoostingClassifier(loss='deviance')),
    #                     ('b', sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier())),
    #                     ('et', sklearn.ensemble.ExtraTreesClassifier()),
    #                     ('rf', sklearn.ensemble.RandomForestClassifier()),
    #                     ('hgb', sklearn.ensemble.HistGradientBoostingClassifier())
    #                 ]),
    #         'Stacking': sklearn.ensemble.StackingClassifier(estimators = [
    #                     ('gb', sklearn.ensemble.GradientBoostingClassifier(loss='deviance')),
    #                     ('ab', sklearn.ensemble.AdaBoostClassifier()),
    #                     ('b', sklearn.ensemble.BaggingClassifier()),
    #                     ('et', sklearn.ensemble.ExtraTreesClassifier()),
    #                     ('rf', sklearn.ensemble.RandomForestClassifier()),
    #                     ('hgb', sklearn.ensemble.HistGradientBoostingClassifier()),
    #                     ('dt', sklearn.tree.DecisionTreeClassifier()),
    #                     ('mlp', sklearn.neural_network.MLPClassifier())
    #                 ]),
    #         'Ref_ab_b_et_rf_hgb_dt_mlp': sklearn.ensemble.StackingClassifier(estimators = [
    #                     ('ab', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.AdaBoostClassifier(), cv=5, method='isotonic')),
    #                     ('b', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.BaggingClassifier(), cv=5, method='isotonic')),
    #                     ('et', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier(), cv=5, method='isotonic')),
    #                     ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
    #                     ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
    #                     ('dt', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
    #                     ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
    #                 ]),
    #         'gb_b_et_rf_hgb': sklearn.ensemble.StackingClassifier(estimators = [
    #                     ('gb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.GradientBoostingClassifier(loss='deviance'), cv=5, method='isotonic')),
    #                     ('b', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.BaggingClassifier(), cv=5, method='isotonic')),
    #                     ('et', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier(), cv=5, method='isotonic')),
    #                     ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
    #                     ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic'))
    #                 ]),
    #         'ab_b_et_rf_hgb_dt_mlp': sklearn.ensemble.StackingClassifier(estimators = [
    #                     ('ab', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.AdaBoostClassifier(), cv=5, method='isotonic')),
    #                     ('b', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.BaggingClassifier(), cv=5, method='isotonic')),
    #                     ('et', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier(), cv=5, method='isotonic')),
    #                     ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
    #                     ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
    #                     ('dt', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
    #                     ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
    #                 ]),
    #         'b_et_rf_hgb_dt_mlp': sklearn.ensemble.StackingClassifier(estimators = [
    #                     ('b', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.BaggingClassifier(), cv=5, method='isotonic')),
    #                     ('et', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier(), cv=5, method='isotonic')),
    #                     ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
    #                     ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
    #                     ('dt', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
    #                     ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
    #                 ]),
    #         'et_rf_hgb_dt_mlp': sklearn.ensemble.StackingClassifier(estimators = [
    #                     ('et', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier(), cv=5, method='isotonic')),
    #                     ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
    #                     ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
    #                     ('dt', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
    #                     ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
    #                 ]),
    #         'rf_hgb_dt_mlp': sklearn.ensemble.StackingClassifier(estimators = [
    #                     ('rf', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
    #                     ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
    #                     ('dt', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
    #                     ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
    #                 ]),
    #         'hgb_dt_mlp': sklearn.ensemble.StackingClassifier(estimators = [
    #                     ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
    #                     ('dt', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
    #                     ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
    #                 ])                      
    #     }
    estimators.update(calibrated_estimators)
    estimators.update(stacking_estimators)
    for k in estimators.keys():
        estimators[k] = SklearnEstimator(estimators[k])
    return estimators
def get_keras_estimators():
    from .model.keras_estimator import KerasEstimator
    estimators = { 
        'relu': KerasEstimator('relu', 100, 100),
        'sigmoid': KerasEstimator('sigmoid', 100, 100),
        'softmax': KerasEstimator('softmax', 100, 100),
        'softplus': KerasEstimator('softplus', 100, 100),
        'softsign': KerasEstimator('softsign', 100, 100),
        'tanh': KerasEstimator('tanh', 100, 100),
        'selu': KerasEstimator('selu', 100, 100),
        'relu': KerasEstimator('relu', 100, 100),
        'softplus': KerasEstimator('softplus', 100, 100),
        'elu': KerasEstimator('elu', 100, 100)
    }
    return estimators
def convert_to_short_name(name):
    return ''.join([s[0].lower() for s in name.split(' ')])
