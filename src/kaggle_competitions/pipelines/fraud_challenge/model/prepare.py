import pandas, logging

def choose_files_encoder(mode, x_train_raw, y_train_raw, x_test_raw, y_test_raw):
    if mode == 'test':
        return split_train_set(x_train_raw, y_train_raw)
    elif mode == 'calibrate':
        return split_train_set(x_train_raw[:100], y_train_raw[:100])
    return x_train_raw, y_train_raw, x_test_raw, y_test_raw
def prepare_y(y):
    return y[['ID', 'fraud_flag']]
def split_train_set(x, y):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    logging.info(f'y={y["fraud_flag"].sum()}')
    logging.info(f'y_train={y_train["fraud_flag"].sum()}')
    logging.info(f'y_test={y_test["fraud_flag"].sum()}')
    return x_train, y_train.astype({'fraud_flag':'float'}), x_test, y_test.astype({'fraud_flag':'float'})

