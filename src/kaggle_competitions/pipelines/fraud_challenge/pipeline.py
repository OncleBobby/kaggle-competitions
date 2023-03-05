from kedro.pipeline import Pipeline, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(\
        [
            # choose_files_node,
            # get_item_encoder_node, 
            # prepare_x_node, 
            # prepare_x_test_node, 
            prepare_y_node, 
            train_model_node, 
            predict_node,
            score_node
        ], \
        tags="fraud_challenge")


