import logging

def choose_files(parameters, x_train_raw, y_train_raw, x_test_raw, y_test_raw):
    mode = parameters['mode'] if 'mode' in parameters else  ''
    test_size = parameters['test_size'] if 'test_size' in parameters else  0.33    
    if mode == 'calibration':
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = split(x_train_raw, y_train_raw, test_size=test_size)
        # x_train, x_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=test_size)
        return format_x(parameters, x_train), y_train, format_x(parameters, x_test), y_test
    return format_x(parameters, x_train_raw), y_train_raw, format_x(parameters, x_test_raw), y_test_raw
def format_x(parameters, x):
    # ID	day	equity	r0	r1	r2	r3	r4	r5	r6	...	r44	r45	r46	r47	r48	r49	r50	r51	r52
    columns = ['ID', 'day', 'equity']
    columns.extend([f'r{i}' for i in range(40, 53)])
    # columns.extend([f'r{i}' for i in range(0, 53)])
    x_formatted = x.fillna(0)
    return x_formatted[columns]
def split(x, y, test_size):
    days = x['day'].unique()
    days.sort()
    split_index = 1-int(len(days)*test_size)
    days_train = days[:split_index]
    days_test = days[split_index:]
    logging.info(f'days={len(days)},days_train={len(days_train)},days_test={len(days_test)}')
    x_train = x[x['day'].isin(days_train)]
    x_test = x[x['day'].isin(days_test)]

    y_train = y[y['ID'].isin(x_train['ID'].unique())]
    y_test = y[y['ID'].isin(x_test['ID'].unique())]

    logging.info(f'x={len(x.index)},y={len(y.index)},x_train={len(x_train.index)},x_test={len(x_test.index)},y_train={len(y_train.index)},y_test={len(y_test.index)}')

    return x_train, x_test, y_train, y_test