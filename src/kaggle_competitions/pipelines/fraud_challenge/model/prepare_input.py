import pandas, logging

# Public functions
def get_encoders(x, y):
    df = pandas.concat([x, y['fraud_flag']], axis=1)
    df = df[df['fraud_flag'] == 1.0]
    logging.info(f'x={len(x)}')
    logging.info(f'y={len(y)}')
    logging.info(f'df={len(df)}')
    encoders = list(_get_encoder(df, 'item'))
    encoders.extend(list(_get_encoder(df, 'make')))
    encoders.extend(list(_get_encoder(df, 'model')))
    return encoders
def prepare_x(x, item_encoder, item_labels, make_encoder, make_labels, model_encoder, model_labels):
    columns = ['ID', 'Nb_of_items']
    x_transformed = _fillna(x)
    logging.info('Prepare X : adding item columns ...')
    x_transformed = _add_columns(\
        x_transformed, item_encoder, item_labels, columns, 'item{0}', \
            {'Nbr_of_prod_purchas{0}': 'Nbr_item{0}', 'cash_price{0}': 'price_item{0}'}).copy()
    return x_transformed[columns]

# Private functions
def _fit_transform(x, encoder):
    x_transformed = x
    for i in range(1, 25):
        column = f'encoded_item{i}'
        x_transformed[column] = encoder.fit_transform(x_transformed[[f'item{i}']].to_numpy().ravel())
    return x_transformed
def _fillna(x):
    values = {f'item{i}' : '' for i in range(1, 25)}
    values.update({f'Nbr_of_prod_purchas{i}' : 0 for i in range(1, 25)})
    values.update({f'cash_price{i}' : 0 for i in range(1, 25)})
    return x.fillna(value=values)
def _get_encoder(x, column_prefix):
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    labels = _get_labels(x, column_prefix)
    encoder.fit(labels)
    return encoder, labels
def _get_labels(x, column_prefix):
    labels = []
    for i in range(1, 25):
        labels.extend(pandas.Series(x[f'{column_prefix}{i}'].values.tolist()).drop_duplicates().tolist())
    labels = pandas.Series(labels).drop_duplicates().tolist()
    return labels
def _add_columns(x, encoder, labels, columns, pattern, column_patterns):
    for label in labels:
        if label == '':
            continue
        encoded_label = encoder.transform([label])[0]
        for column_pattern in column_patterns.values():
            encoded_column = column_pattern.format(encoded_label)
            x[encoded_column] = 0
            columns.append(encoded_column)
    return _update_columns(x.copy(), encoder, pattern, column_patterns, set(labels))
def _update_columns(x, item_encoder, pattern, column_patterns, labels):
    def update_columns(row):
        for i in range(1, 25):
            column = pattern.format(i)
            label = row[column]
            if label == '':
                continue
            encoded_label = int(item_encoder.transform([label])[0]) if label in labels else 0 
            for k, v in column_patterns.items():
                encoded_column = v.format(encoded_label)
                row[encoded_column] = row[k.format(i)] + (row[encoded_column] if encoded_column in row else 0)
        return row
    x = x.apply(update_columns, axis=1)
    return x