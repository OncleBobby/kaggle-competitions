from kedro.pipeline import node
from .model.scope import *
from .model.predict import *
from .model.prepare import *
from .model.prepare_input import *
from .calibration import calibrate_model as calibrate

choose_files_node = node(func=choose_files_encoder, \
    inputs=['params:parameters', 'x_train_raw', 'y_train_raw', 'x_test_raw', 'y_test_raw'], \
    outputs=['x_train', 'y_train', 'x_test', 'y_test'], tags=['preparation'])
get_encoders_node = node(func=get_encoders, inputs=['x_train', 'y_train'], \
    outputs=['item_encoder', 'item_labels', 'make_encoder', 'make_labels', 'model_encoder', 'model_labels'], tags=['preparation'])
prepare_x_node = node(func=prepare_x, \
    inputs=['x_train', 'item_encoder', 'item_labels', 'make_encoder', 'make_labels', 'model_encoder', 'model_labels'],\
    outputs='x_train_transformed', tags=['preparation'])
prepare_x_test_node = node(func=prepare_x, \
    inputs=['x_test', 'item_encoder', 'item_labels', 'make_encoder', 'make_labels', 'model_encoder', 'model_labels'],\
    outputs='x_test_transformed', tags=['preparation'])
prepare_y_node = node(func=prepare_y, inputs=['y_train'], outputs='y_train_transformed', tags=['preparation'])
train_model_node = node(func=train_model, inputs=['x_train_transformed', 'y_train_transformed'], outputs='model')
predict_node = node(func=predict, inputs=['model', 'x_test_transformed'], outputs='y_prediction')
calibrate_model_node = node(func=calibrate, \
    inputs=['params:parameters', 'x_train_transformed', 'y_train_transformed', 'x_test_transformed', 'y_test'], \
    outputs=['model_fraud', 'model_score_fraud'], tags=['calibration'])