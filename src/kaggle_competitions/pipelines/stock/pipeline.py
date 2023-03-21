from kedro.pipeline import Pipeline, pipeline
from .nodes import *

def create_pipeline() -> Pipeline:
    return pipeline(\
        [
            format_x_training_node,
            format_x_test_node,
            format_y_training_node,
            split_train_dataset_node,
            train_select_features_node,
            test_select_features_node,
            calibrate_model_node,
            to_submit_select_features_node,
            predict_node
        ], \
        tags="stock")