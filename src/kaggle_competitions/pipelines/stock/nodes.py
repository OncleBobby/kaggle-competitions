from kedro.pipeline import node
import logging, pandas

def choose_files(mode, x_train_raw, y_train_raw, x_test_raw, y_test_raw):
    if mode == 'test':
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=0.33)
        return x_train, y_train, x_test, y_test
    return x_train_raw, y_train_raw, x_test_raw, y_test_raw
def train_model(x, y):
    # from sklearn.ensemble import GradientBoostingClassifier as Model # 0.4974936127550855
    # from sklearn.ensemble import AdaBoostClassifier as Model # 0.4673666584018772
    # from sklearn.ensemble import BaggingClassifier as Model # 0.44524576968547086
    # from sklearn.ensemble import ExtraTreesClassifier as Model # 0.49066617796607126
    # from sklearn.ensemble import RandomForestClassifier as Model # 0.48460772793750384
    # from sklearn.ensemble import StackingClassifier as Model
    # from sklearn.ensemble import HistGradientBoostingClassifier as Model # 0.5281847288250704
    # from sklearn.tree import DecisionTreeClassifier as Model # 0.40449676415524866
    # from sklearn.tree import ExtraTreeClassifier as Model # 0.3991641782463554
    # from sklearn.neural_network import MLPClassifier as Model # 0.41130982539733874
    # from sklearn.neighbors import KNeighborsClassifier as Model # 0.37373018696391164
    # from sklearn.calibration import CalibratedClassifierCV as Model # 0.41128107830349026
    # from sklearn.gaussian_process import GaussianProcessClassifier as Model
    # model = sklearn.ensemble.RandomForestClassifier()
    # model = Model()

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.ensemble import StackingClassifier as Model # 0.21714637918918994
    estimators = [
            ('gbc', GradientBoostingClassifier()),
            ('abc', AdaBoostClassifier()),
            ('bc', BaggingClassifier()),
            ('etc', ExtraTreesClassifier()),
            ('rfc', RandomForestClassifier()),
            ('dtc', DecisionTreeClassifier()),
            ('hgb', HistGradientBoostingClassifier())
        ]
    model= Model(estimators=estimators)

    model.fit(x.fillna(0), y[['reod']])

    logging.info(f'model={type(model)}')

    return model
def predict(model, x):
    y_predict =  model.predict(x.fillna(0))
    df = pandas.DataFrame(x['ID'].copy(), columns=['ID'])
    df['reod'] = y_predict
    return df
def score(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_true['reod'].values.tolist(), y_pred['reod'].values.tolist())
    logging.info(f'score={score}')
    return score

choose_files_node = node(func=choose_files, inputs=['params:mode', 'x_train_raw', 'y_train_raw', 'x_test_raw', 'y_test_raw'],  \
                            outputs=['x_train_stock', 'y_train_stock', 'x_test_stock', 'y_test_stock'])
train_model_node = node(func=train_model, inputs=['x_train_stock', 'y_train_stock'], outputs='model_stock')
predict_node = node(func=predict, inputs=['model_stock', 'x_test_stock'], outputs='y_prediction_stock')
score_node = node(func=score, inputs=['y_test_stock', 'y_prediction_stock'], outputs='score_stock')
