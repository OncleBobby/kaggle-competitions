from kedro.pipeline import Pipeline, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(\
        [
            choose_files_node,
            get_encoders_node, 
            prepare_x_node, 
            prepare_x_test_node, 
            prepare_y_node, 
            # train_model_node, 
            # predict_node,
            calibrate_model_node
        ], \
        tags="fraud_challenge")


