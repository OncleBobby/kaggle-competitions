import pandas, logging

# Public functions
def get_item_encoder(x1, x2):
    return _get_encoder(pandas.concat([x1, x2]), 'item')
def prepare_x(x, item_encoder, item_labels):
    column_patterns = {'Nbr_of_prod_purchas{0}': 'Nbr_of_prod_purchas_encoded{0}', 'cash_price{0}': 'cash_price_encoded{0}'}
    columns = ['ID', 'Nb_of_items']
    x_transformed = _fillna(x)
    # for i in range(1, 25):
    #     for column_pattern in column_patterns.keys():
    #         columns.append(column_pattern.format(i))
    for i, label in enumerate(item_labels):
        if label == '':
            continue
        encoded_label = item_encoder.transform([label])[0]
        for column_pattern in column_patterns.values():
            encoded_column = column_pattern.format(encoded_label)
            x_transformed[encoded_column] = 0
            columns.append(encoded_column)

    x_transformed = _update_columns(x_transformed, item_encoder, column_patterns)
        # print(f'adding column for {label} -> {encoded_label} ({i+1}/{len(item_labels)}) ...')            
        # logging.info(f'adding column for {label} -> {encoded_label} ({i+1}/{len(item_labels)}) ...')
        # x_transformed = _add_column(x_transformed, label, encoded_column)
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
    print(f'Nbr item labels={len(labels)}')
    return labels
def _update_columns(x, item_encoder, column_patterns):
    def update_columns(row):
        for i in range(1, 25):
            column = f'item{i}'
            label = row[column]
            if label == '':
                continue
            encoded_label = int(item_encoder.transform([label])[0])
            # print(f'{column}={label} -> {encoded_label}')
            for k, v in column_patterns.items():
                # print(f'\t{row[k.format(i)]}: {k.format(i)} -> {v.format(encoded_label)}')
                row[v.format(encoded_label)] = row[k.format(i)]

            # if row[column] == label:
            #     return row[f'Nbr_of_prod_purchas{i}']
        return row
    x = x.apply(update_columns, axis=1)
    return x