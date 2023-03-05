from kedro.pipeline import node
from .model.scope import *
from .model.predict import *
from .model.prepare import *
from .model.prepare_input import *

choose_files_node = node(func=choose_files_encoder, \
                         inputs=['params:mode', 'x_train_raw', 'y_train_raw', 'x_test_raw', 'y_test_raw'], \
                         outputs=['x_train', 'y_train', 'x_test', 'y_test'])

get_item_encoder_node = node(func=get_item_encoder, inputs=['x_train', 'x_test'], outputs=['item_encoder', 'item_labels'])

prepare_x_node = node(func=prepare_x, inputs=['x_train', 'item_encoder', 'item_labels'], outputs='x_train_transformed')
prepare_x_test_node = node(func=prepare_x, inputs=['x_test', 'item_encoder', 'item_labels'], outputs='x_test_transformed')

prepare_y_node = node(func=prepare_y, inputs=['y_train'], outputs='y_train_transformed')

train_model_node = node(func=train_decision_tree_regressor, inputs=['x_train_transformed', 'y_train_transformed'], outputs='model')

predict_node = node(func=predict, inputs=['model', 'x_test_transformed'], outputs='y_prediction')

score_node = node(func=score, inputs=['y_test', 'y_prediction'], outputs='score')
