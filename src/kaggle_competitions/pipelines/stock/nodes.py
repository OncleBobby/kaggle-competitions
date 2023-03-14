from kedro.pipeline import node
import logging, pandas, time, datetime
from ...calibration import calibrate_model, predict, train

def choose_files(parameters, x_train_raw, y_train_raw, x_test_raw, y_test_raw):
    mode = parameters['mode'] if 'mode' in parameters else  ''
    test_size = parameters['test_size'] if 'test_size' in parameters else  0.33    
    if mode == 'calibration':
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=test_size)
        return x_train.fillna(0), y_train, x_test.fillna(0), y_test
    return x_train_raw.fillna(0), y_train_raw, x_test_raw.fillna(0), y_test_raw

choose_files_node = node(func=choose_files, inputs=['params:parameters', 'x_train_raw', 'y_train_raw', 'x_test_raw', 'y_test_raw'],  \
                        outputs=['x_train_stock', 'y_train_stock', 'x_test_stock', 'y_test_stock'], tags=['calibration', 'submission'])
train_node = node(func=train, inputs=['params:parameters', 'x_train_stock', 'y_train_stock'], outputs='model_stock', tags='submission')
predict_node = node(func=predict, inputs=['params:parameters', 'model_stock', 'x_test_stock'], outputs='y_prediction_stock', tags='submission')
calibrate_model_node = node(func=calibrate_model, inputs=['params:parameters', 'x_train_stock', 'y_train_stock', 'x_test_stock', 'y_test_stock'], \
        outputs='model_score', tags=['calibration'])
