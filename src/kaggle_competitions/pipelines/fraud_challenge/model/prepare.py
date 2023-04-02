import pandas, logging

def choose_files_encoder(parameters, x_train_raw, y_train_raw, x_test_raw, y_test_raw):
    mode = parameters['mode']
    test_size = parameters['test_size'] if 'test_size' in parameters else  0.33    
    if mode == 'calibration':
        return split_train_set(x_train_raw, y_train_raw, test_size)
    return x_train_raw, y_train_raw, x_test_raw, y_test_raw
def prepare_y(y):
    return pandas.concat([y, pandas.get_dummies(y['fraud_flag'])], axis=1)
    # return y[['ID', 'fraud_flag']]
def split_train_set(x, y, test_size):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, y_train.astype({'fraud_flag':'float'}), x_test, y_test.astype({'fraud_flag':'float'})

