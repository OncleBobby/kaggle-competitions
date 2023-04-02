
import keras, pandas, logging
from .estimator import Estimator
class KerasEstimator(Estimator):
    def __init__(self, activation_name='relu', batch_size=10, epochs=10):
        self.activation_name = activation_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
    def fit(self, x, y):
        self.model = self.build_model(x, y, self.activation_name, self.batch_size, self.epochs)
    def predict(self, x):
        pass
    def predict_proba(self, x):
        y = pandas.DataFrame(x['ID'].copy(), columns=['ID'])
        prediction = self.model.predict(x)
        prediction = self.model.predict(x)
        y['fraud_flag'] = [p[0] for p in prediction]
        y = y.reset_index(drop=True)
        return y  
    def build_model(self, x, y, activation_name='relu', batch_size=10, epochs=10):
        # y_columns = y.columns
        y_columns = ['0.0', '1.0']
        inputs = keras.Input(shape=x.shape[1])
        hidden_layer = keras.layers.Dense(20, activation=activation_name)(inputs)
        output_layer = keras.layers.Dense(len(y_columns), activation="softmax")(hidden_layer)
        model = keras.Model(inputs=inputs, outputs=output_layer)
        model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
        history = model.fit(x, y[y_columns], batch_size = batch_size, epochs = epochs, verbose=0)
        return model