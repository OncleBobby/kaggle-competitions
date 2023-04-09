import pandas, logging

def prepare_y(y):
    return pandas.concat([y, pandas.get_dummies(y['fraud_flag'])], axis=1)
def split_train_set(x, y, parameters):
    from sklearn.model_selection import train_test_split
    test_size = parameters['test_size'] if 'test_size' in parameters else  0.33    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, y_train.astype({'fraud_flag':'float'}), x_test, y_test.astype({'fraud_flag':'float'})

