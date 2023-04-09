from kedro.pipeline import Pipeline, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(\
        [
            get_encoders_node,
            prepare_x_node,
            prepare_x_submission_node,
            prepare_y_node,
            split_train_set_node,
            calibrate_model_node,
            train_model_node,
            predict_node
        ], \
        tags="fraud_challenge")


