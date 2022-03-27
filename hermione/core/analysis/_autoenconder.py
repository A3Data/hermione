import keras


class Autoencoder:
    def __init__(self, n_components, n_layers=1, **kwargs):
        self.n_components = n_components
        self.n_layers = n_layers
        self.kwargs = kwargs

    def fit(self, X, y=None):
        input_ = keras.layers.Input(shape=(X.shape[1]))
        encoded = keras.layers.Dense(self.n_components, activation="relu")(input_)
        decoded = keras.layers.Dense(X.shape[1], activation="relu")(encoded)

        self.autoencoder = keras.Model(input_, decoded)
        self.encoder = keras.Model(input_, encoded)
        self.autoencoder.compile(loss=keras.losses.MeanSquaredError())
        print(X.shape[1])
        self.autoencoder.fit(X, X, epochs=100, batch_size=64, shuffle=True)

    def transform(self, X, y=None):
        return self.encoder.predict(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.encoder.predict(X)
