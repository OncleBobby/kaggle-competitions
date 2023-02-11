import numpy as np, logging

def prepare_x(x):
    columns = ['ID', 'Nb_of_items']
    for i in range(1, 25):
        columns.append(f'Nbr_of_prod_purchas{i}')
    return x[columns].reset_index(drop=True)
def prepare_y(y):
    return y[['ID', 'fraud_flag']].reset_index(drop=True)