from kedro.pipeline import node
from .calibration import calibrate_model, predict
from .preparation import format_x, format_y, select_features, split_train_dataset

format_x_training_node = node(func=format_x, inputs= 'input_training_stock', outputs='input_training_stock_formatted', tags=['preparation'])
format_x_test_node = node(func=format_x, inputs='input_test_stock', outputs='input_test_stock_formatted', tags=['preparation'])
format_y_training_node = node(func=format_y, inputs='output_training_stock', outputs='output_training_stock_formatted', tags=['preparation', 'calibration', 'submission'])
split_train_dataset_node = node(func=split_train_dataset, inputs=['params:parameters', 'input_training_stock_formatted', 'output_training_stock_formatted'],  \
                        outputs=['x_train_split_stock', 'x_test_split_stock', 'y_train_stock', 'y_test_stock'], tags=['preparation', 'calibration'])
train_select_features_node = node(func=select_features, inputs=['params:parameters', 'x_train_split_stock'], outputs='x_train_stock', tags=['preparation', 'calibration'])
test_select_features_node = node(func=select_features, inputs=['params:parameters', 'x_test_split_stock'], outputs='x_test_stock', tags=['preparation', 'calibration'])
calibrate_model_node = node(func=calibrate_model, inputs=['params:parameters', 'x_train_stock', 'y_train_stock', 'x_test_stock', 'y_test_stock'], \
        outputs=['model_stock', 'model_score'], tags=['calibration'])
to_submit_select_features_node = node(func=select_features, inputs=['params:parameters', 'input_test_stock_formatted'], outputs='x_to_submit_stock', tags=['preparation', 'submission'])
predict_node = node(func=predict, inputs=['params:parameters', 'model_stock', 'x_to_submit_stock'], outputs='y_prediction_stock', tags=['submission'])

