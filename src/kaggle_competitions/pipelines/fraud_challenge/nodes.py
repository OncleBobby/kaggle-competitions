from kedro.pipeline import node
from .model.scope import *
from .model.predict import *
from .model.prepare import *

prepare_x_node = node(func=prepare_x, inputs=['x_train'], outputs='x_train2')
prepare_y_node = node(func=prepare_y, inputs=['y_train'], outputs='y_train2')
train_model_node = node(func=train_decision_tree_regressor, inputs=['x_train2', 'y_train2'], outputs='model')
prepare_x_test_node = node(func=prepare_x, inputs=['x_test'], outputs='x_test2')
predict_node = node(func=predict, inputs=['model', 'x_test2'], outputs='y_prediction')
