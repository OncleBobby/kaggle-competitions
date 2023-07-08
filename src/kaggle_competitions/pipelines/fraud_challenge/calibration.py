import pandas, logging, datetime, numpy, itertools, pickle, os
from sklearn.metrics import average_precision_score

def calibrate_model(x_train, y_train, x_test, y_test, root_folder):

    model_scores_filename = os.path.join(root_folder, f'data/fraud/07_model_output/model_scores.csv')
    model_score_fraud = pandas.read_csv(model_scores_filename, sep=',').set_index('index')

    lines = [r.to_dict() for i, r in model_score_fraud.iterrows()]
    already_estimated = {r['name']: r for i, r in model_score_fraud.iterrows()}

    best_score = 0
    best_model = None
    best_name = None
    i = 0
    estimators = get_estimators()
    nbr = len(estimators.keys())
    # for name in params['estimator_names']:
    for name, estimator in estimators.items():
        i = i + 1
        if name in already_estimated:
            line = already_estimated[name]
            line = already_estimated[name]
            current_score = line['score']
            duration = line["duration"]
            logging.info(f'{i}/{nbr} - {current_score}\t{name} in {duration} - already estimated and would be skip')
        else:
            start_time = datetime.datetime.now()

            saved_estimator = _load_if_exist(name, root_folder)
            if saved_estimator is None:
                estimator.fit(x_train.fillna(0), y_train)
                y_predict = estimator.predict_proba(x_test.fillna(0))
            else:
                logging.info(f'{i}/{nbr} - {name} - using previous fitted estimator')
                y_predict = saved_estimator.predict_proba(x_test.fillna(0))
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
            _save_model(name, estimator, root_folder)
            pandas.DataFrame(lines).to_csv(model_scores_filename, sep=',')
            logging.info(f'{i}/{nbr} - {current_score}\t{name} in {duration}')
        if current_score > best_score:
            best_score = current_score
            best_name = name
            best_model = _load_if_exist(name, root_folder)
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
            # 'dmf': sklearn.dummy.DummyClassifier(strategy='most_frequent'),
            # 'dp': sklearn.dummy.DummyClassifier(strategy='prior'),
            # 'ds': sklearn.dummy.DummyClassifier(strategy='stratified'),
            # 'du': sklearn.dummy.DummyClassifier(strategy='uniform'),
            # 'dc': sklearn.dummy.DummyClassifier(strategy='constant', constant=0),
            'xgb': xgboost.XGBClassifier(),
            'rf': sklearn.ensemble.RandomForestClassifier(),
            # 'gbl': sklearn.ensemble.GradientBoostingClassifier(loss='log_loss'),
            # 'gbd': sklearn.ensemble.GradientBoostingClassifier(loss='deviance'),
            # 'gbe': sklearn.ensemble.GradientBoostingClassifier(loss='exponential'),
            'gb': sklearn.ensemble.GradientBoostingClassifier(),
            'ab': sklearn.ensemble.AdaBoostClassifier(),
            'b': sklearn.ensemble.BaggingClassifier(),
            # 'bet': sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.ExtraTreesClassifier()),
            # 'brf': sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.RandomForestClassifier()),
            # 'bhgb': sklearn.ensemble.BaggingClassifier(estimator=sklearn.ensemble.HistGradientBoostingClassifier()),
            # 'et': sklearn.ensemble.ExtraTreesClassifier(),
            # 'dt': sklearn.tree.DecisionTreeClassifier(),
            # 'hgb': sklearn.ensemble.HistGradientBoostingClassifier(),
            # 'hgbl': sklearn.ensemble.HistGradientBoostingClassifier(loss='log_loss'),
            'hgba': sklearn.ensemble.HistGradientBoostingClassifier(loss='auto'),
            # 'm': sklearn.neural_network.MLPClassifier(),
            'k': sklearn.neighbors.KNeighborsClassifier()           
        }
    estimators = {}
    estimators.update(basic_estimators)
    calibrated_estimators = {}
    for name, estimator in basic_estimators.items():
        calibrated_estimators[f"{name}c"] = sklearn.calibration.CalibratedClassifierCV(estimator)   
    stacking_estimators = {}
    for names in itertools.permutations(basic_estimators.keys(), 4):
        names = list(names)
        names.sort()
        stacking_estimators['_'.join(names)] = sklearn.ensemble.StackingClassifier(estimators = [
                        (name, sklearn.calibration.CalibratedClassifierCV(basic_estimators[name], cv=5, method='isotonic'))
                        for name in names
                    ])
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
def _save_model(name, model, root_folder):
    model_folder = os.path.join(root_folder, f'./data/fraud/06_models/')
    filename = os.path.join(model_folder, f'{name}.pkl')
    pickle.dump(model, open(filename, 'wb'))
def _load_if_exist(name, root_folder):
    model_folder = os.path.join(root_folder, f'./data/fraud/06_models/')    
    filename = os.path.join(model_folder, f'{name}.pkl')
    if os.path.exists(filename):
        return pickle.load(open(filename, 'rb'))
    return None