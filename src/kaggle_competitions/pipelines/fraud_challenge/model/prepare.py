import numpy as np, logging

def prepare_x(x):
    columns = ['ID', 'Nb_of_items']
    for i in range(1, 25):
        columns.append(f'Nbr_of_prod_purchas{i}')
        # print(f'prepare_x={x[columns].fillna(0).tail}')
    return x[columns].fillna(0)
def prepare_y(y):
    # print(f"prepare_y={y[['ID', 'fraud_flag']].tail}")
    return y[['ID', 'fraud_flag']]
def split_train_set(x, y):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(prepare_x(x), prepare_y(y), test_size=0.33)
    logging.info(f'y={y["fraud_flag"].sum()}')
    logging.info(f'y_train={y_train["fraud_flag"].sum()}')
    logging.info(f'y_test={y_test["fraud_flag"].sum()}')
    return x_train, x_test, y_train, y_test

