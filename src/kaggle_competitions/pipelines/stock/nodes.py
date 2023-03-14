from kedro.pipeline import node
import logging, pandas, time, datetime
from ...calibration import calibrate_model, get_estimators

def choose_files(dataset_parameters, x_train_raw, y_train_raw, x_test_raw, y_test_raw):
    mode = dataset_parameters['mode'] if 'mode' in dataset_parameters else  ''
    test_size = dataset_parameters['test_size'] if 'test_size' in dataset_parameters else  0.33    
    if mode == 'calibration':
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=test_size)
        return x_train.fillna(0), y_train, x_test.fillna(0), y_test
    return x_train_raw.fillna(0), y_train_raw, x_test_raw.fillna(0), y_test_raw
def train_model(estimator_name, estimators, x, y):
    start_time = datetime.datetime.now()
    model = estimators[estimator_name]
    model.fit(x.fillna(0), y[['reod']])
    end_time = datetime.datetime.now()
    duration = (end_time - start_time)
    logging.info(f'train_model done in {duration}')
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

choose_files_node = node(func=choose_files, inputs=['params:parameters', 'x_train_raw', 'y_train_raw', 'x_test_raw', 'y_test_raw'],  \
                            outputs=['x_train_stock', 'y_train_stock', 'x_test_stock', 'y_test_stock'], tags=['calibration', 'submission'])
train_model_node = node(func=train_model, inputs=['params:estimator_name', 'estimators', 'x_train_stock', 'y_train_stock'], \
                        outputs='model_stock', tags='submission')
predict_node = node(func=predict, inputs=['model_stock', 'x_test_stock'], outputs='y_prediction_stock', tags='submission')
get_estimators_node = node(func=get_estimators, inputs=['params:estimator_names'], outputs='estimators', \
                            tags=['calibration', 'submission'])
calibrate_model_node = node(func=calibrate_model, \
    inputs=['estimators', 'params:parameters', 'x_train_stock', 'y_train_stock', 'x_test_stock', 'y_test_stock'], \
        outputs='model_score', tags=['calibration'])

