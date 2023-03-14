def choose_files(parameters, x_train_raw, y_train_raw, x_test_raw, y_test_raw):
    mode = parameters['mode'] if 'mode' in parameters else  ''
    test_size = parameters['test_size'] if 'test_size' in parameters else  0.33    
    if mode == 'calibration':
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=test_size)
        return x_train.fillna(0), y_train, x_test.fillna(0), y_test
    return x_train_raw.fillna(0), y_train_raw, x_test_raw.fillna(0), y_test_raw