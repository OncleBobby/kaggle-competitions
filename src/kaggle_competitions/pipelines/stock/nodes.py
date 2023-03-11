from kedro.pipeline import node
import logging, pandas, time, datetime
from ...calibration import calibrate_model, get_estimators

def choose_files(mode, x_train_raw, y_train_raw, x_test_raw, y_test_raw):
    if mode == 'test':
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=0.33)
        return x_train.fillna(0), y_train, x_test.fillna(0), y_test
    if mode == 'calibrate':
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=0.99)
        return x_train.fillna(0), y_train, x_test.fillna(0), y_test
    return x_train_raw.fillna(0), y_train_raw, x_test_raw.fillna(0), y_test_raw
def train_model(x, y):
    start_time = time.time()
    import sklearn.ensemble
    import sklearn.tree
    import sklearn.neural_network
    import sklearn.calibration
    model = sklearn.ensemble.StackingClassifier(estimators = [
                ('gbc', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.GradientBoostingClassifier(loss='deviance'), cv=5, method='isotonic')),
                ('abc', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.AdaBoostClassifier(), cv=5, method='isotonic')),
                ('bc', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.BaggingClassifier(), cv=5, method='isotonic')),
                ('etc', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.ExtraTreesClassifier(), cv=5, method='isotonic')),
                ('rfc', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.RandomForestClassifier(), cv=5, method='isotonic')),
                ('hgb', sklearn.calibration.CalibratedClassifierCV(sklearn.ensemble.HistGradientBoostingClassifier(), cv=5, method='isotonic')),
                ('dtc', sklearn.calibration.CalibratedClassifierCV(sklearn.tree.DecisionTreeClassifier(), cv=5, method='isotonic')),
                ('mlp', sklearn.calibration.CalibratedClassifierCV(sklearn.neural_network.MLPClassifier(), cv=5, method='isotonic'))
            ])
    model.fit(x.fillna(0), y[['reod']])
    logging.info(f'train_model done in {time.time() - start_time}s {datetime.datetime.fromtimestamp(time.time() - start_time).strftime("%M:%S.%f")}')
    return model
def predict(model, x):
    y_predict =  model.predict(x.fillna(0))
    df = pandas.DataFrame(x['ID'].copy(), columns=['ID'])
    df['reod'] = y_predict
    return df
def score(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_true['reod'].values.tolist(), y_pred['reod'].values.tolist())
    logging.info(f'score={score}')
    return score


choose_files_node = node(func=choose_files, inputs=['params:mode', 'x_train_raw', 'y_train_raw', 'x_test_raw', 'y_test_raw'],  \
                            outputs=['x_train_stock', 'y_train_stock', 'x_test_stock', 'y_test_stock'])
train_model_node = node(func=train_model, inputs=['x_train_stock', 'y_train_stock'], outputs='model_stock')
predict_node = node(func=predict, inputs=['model_stock', 'x_test_stock'], outputs='y_prediction_stock')
score_node = node(func=score, inputs=['y_test_stock', 'y_prediction_stock'], outputs='score_stock')
get_estimators_node = node(func=get_estimators, inputs=['params:estimator_names'], outputs='estimators')
calibrate_model_node = node(func=calibrate_model, \
    inputs=['estimators', 'params:target_field', 'params:id_field', 'x_train_stock', 'y_train_stock', 'x_test_stock', 'y_test_stock'], \
        outputs='model_score')

