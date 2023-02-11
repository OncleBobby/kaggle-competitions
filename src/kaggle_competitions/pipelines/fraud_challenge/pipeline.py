from kedro.pipeline import Pipeline, pipeline
from .nodes import *

def create_pipeline() -> Pipeline:
    return pipeline([prepare_x_node, prepare_y_node, predict_benchmark_node, score_node], tags="fraud_challenge")
