from kedro.pipeline import Pipeline, pipeline
from .nodes import *

def create_pipeline() -> Pipeline:
    return pipeline(\
        [
            choose_files_node,
            train_node, 
            predict_node,
            calibrate_model_node
        ], \
        tags="stock")

