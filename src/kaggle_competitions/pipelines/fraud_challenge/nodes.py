from kedro.pipeline import node
from .model.scope import *
from .model.predict import *
from .model.prepare import *


prepare_x_node = node(func=prepare_x, inputs=['x_train'], outputs="x", name="prepare_x")
prepare_y_node = node(func=prepare_y, inputs=['y_train'], outputs="y", name="prepare_y")

predict_benchmark_node = node(func=predict_benchmark, inputs=['x'], outputs="y_pred_proba", name="predict_benchmark")
score_node = node(func=score, inputs=['y', 'y_pred_proba'], outputs="score_node", name="score")
