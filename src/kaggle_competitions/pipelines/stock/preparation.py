def choose_files(parameters, x_train_raw, y_train_raw, x_test_raw, y_test_raw):
    mode = parameters['mode'] if 'mode' in parameters else  ''
    test_size = parameters['test_size'] if 'test_size' in parameters else  0.33    
    if mode == 'calibration':
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=test_size)
        return format_x(parameters, x_train), y_train, format_x(parameters, x_test), y_test
    return format_x(parameters, x_train_raw), y_train_raw, format_x(parameters, x_test_raw), y_test_raw
def format_x(parameters, x):
    # ID	day	equity	r0	r1	r2	r3	r4	r5	r6	...	r44	r45	r46	r47	r48	r49	r50	r51	r52
    columns = ['ID', 'day', 'equity']
    columns.extend([f'r{i}' for i in range(40, 53)])
    # columns.extend([f'r{i}' for i in range(0, 53)])
    x_formatted = x.fillna(0)
    return x_formatted[columns] 