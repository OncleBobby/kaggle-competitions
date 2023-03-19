import logging

def format_x(x):
    # ID	day	equity	r0	r1	r2	r3	r4	r5	r6	...	r44	r45	r46	r47	r48	r49	r50	r51	r52
    x_formatted = x.fillna(0)
    return _add_columns(x_formatted)
def format_y(y):
    return y.fillna(0)
def select_features(parameters, x):
    # ID	day	equity	r0	r1	r2	r3	r4	r5	r6	...	r44	r45	r46	r47	r48	r49	r50	r51	r52
    columns = ['ID', 'day', 'equity', 'nbr_positif', 'nbr_negatif']
    columns.extend([f'r{i}' for i in range(40, 53)])
    # columns.extend([f'r{i}' for i in range(0, 53)])
    return x[columns]
def prepare(parameters, input_training, output_training, input_test, output_test):
    return format_x(parameters, input_training), output_training.reset_index(drop=True), format_x(parameters, input_test), output_test.reset_index(drop=True)
def split_train_dataset(parameters, x_train_raw, y_train_raw):
    test_size = parameters['test_size'] if 'test_size' in parameters else  0.33    
    return split(x_train_raw, y_train_raw, test_size=test_size)
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
def _add_columns(x):
    logging.info('Adding columns ...')
    def update_columns(row):
        nbr_positif = 0
        nbr_negatif = 0
        for j in range(0, 53):
            if row[j] > 0:
                nbr_positif = nbr_positif + 1
            if row[j] < 0:
                nbr_negatif = nbr_negatif + 1
        row['nbr_positif'] = nbr_positif
        row['nbr_negatif'] = nbr_negatif
        return row
    x['nbr_positif'] = 0
    x['nbr_negatif'] = 0
    x = x.apply(update_columns, axis=1)
    logging.info('Adding columns done !')
    return x