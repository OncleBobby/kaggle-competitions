from kedro.pipeline import node
from .calibration import calibrate_model, predict, train
from .preparation import choose_files

choose_files_node = node(func=choose_files, inputs=['params:parameters', 'x_train_raw', 'y_train_raw', 'x_test_raw', 'y_test_raw'],  \
                        outputs=['x_train_stock', 'y_train_stock', 'x_test_stock', 'y_test_stock'], tags=['calibration', 'submission'])
train_node = node(func=train, inputs=['params:parameters', 'x_train_stock', 'y_train_stock'], outputs='model_stock', tags='submission')
predict_node = node(func=predict, inputs=['params:parameters', 'model_stock', 'x_test_stock'], outputs='y_prediction_stock', tags='submission')
calibrate_model_node = node(func=calibrate_model, inputs=['params:parameters', 'x_train_stock', 'y_train_stock', 'x_test_stock', 'y_test_stock'], \
        outputs='model_score', tags=['calibration'])
