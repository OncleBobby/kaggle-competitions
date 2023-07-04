from kedro.pipeline import node
from .model.scope import *
from .model.predict import *
from .model.prepare import *
from .model.prepare_input import *
from .calibration import calibrate_model as calibrate

get_encoders_node = node(func=get_encoders, inputs=['x_raw', 'y_raw'], \
    outputs=['item_encoder', 'item_labels', 'make_encoder', 'make_labels', 'model_encoder', 'model_labels'], tags=['preparation'])
prepare_x_node = node(func=prepare_x, \
    inputs=['x_raw', 'item_encoder', 'item_labels', 'make_encoder', 'make_labels', 'model_encoder', 'model_labels'],\
    outputs='x_model', tags=['preparation'])
prepare_y_node = node(func=prepare_y, inputs=['y_raw'], outputs='y_model', tags=['preparation'])
split_train_set_node = node(func=split_train_set, inputs=['x_model', 'y_model', 'params:parameters'], \
    outputs=['x_train', 'y_train', 'x_test', 'y_test'], tags=['preparation'])
calibrate_model_node = node(func=calibrate, inputs=['params:parameters', 'x_train', 'y_train', 'x_test', 'y_test'], \
    outputs=['best_model_fraud', 'best_model_fraud_name', 'model_score_fraud'], tags=['calibration'])
prepare_x_submission_node = node(func=prepare_x, \
    inputs=['x_submission_raw', 'item_encoder', 'item_labels', 'make_encoder', 'make_labels', 'model_encoder', 'model_labels'],\
    outputs='x_model_submission', tags=['preparation'])
train_model_node = node(func=train_model, inputs=['x_model', 'y_model', 'params:parameters'], outputs='submission_model', tags=['submission'])
predict_node = node(func=predict, inputs=['submission_model', 'x_model_submission'], outputs='y_submission', tags=['submission'])
predict_with_best_model_node = node(func=predict, inputs=['best_model_fraud', 'x_model_submission'], outputs='y_predict_with_best_model', tags=['calibration'])
