import numpy as np
import random
import tensorflow as tf
from sklearn.base import ClassifierMixin
from tensorflow import keras


class GeneralizedMatrixFactorization(ClassifierMixin):
    def __init__(self,
                 n_factors=8,
                 epochs=10,
                 seed=None,
                 user_input=None,
                 item_input=None,
                 user_preprocessing_layers=None,
                 item_preprocessing_layers=None):
        self.n_factors = n_factors
        self.epochs = epochs
        self.seed = seed
        self.user_input = user_input
        self.item_input = item_input
        self.user_preprocessing_layers = user_preprocessing_layers
        self.item_preprocessing_layers = item_preprocessing_layers

    def create_model(self):
        user_input = (self.user_input
                      if self.user_input is not None else
                      keras.Input(shape=(1), name="user", dtype="int64"))
        item_input = (self.item_input
                      if self.item_input is not None else
                      keras.Input(shape=(1), name="item", dtype="int64"))

        user_preprocessing_layers = (
            self.user_preprocessing_layers
            if self.user_preprocessing_layers is not None
            else user_input
        )
        item_preprocessing_layers = (
            self.item_preprocessing_layers
            if self.item_preprocessing_layers is not None
            else item_input
        )

        gmf_output = [
            keras.layers.Dense(self.n_factors)(user_preprocessing_layers),
            keras.layers.Dense(self.n_factors)(item_preprocessing_layers)
        ]
        gmf_output = keras.layers.Multiply()(gmf_output)
        gmf_output = keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_constraint=keras.constraints.unit_norm()
        )(gmf_output)

        return keras.Model(inputs=[user_input, item_input],
                           outputs=[gmf_output],
                           name="generalized_matrix_factorization")

    def fit(self, X=None, y=None):
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)

        self.model = self.create_model()
        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss=keras.losses.BinaryCrossentropy(),
                           metrics=[keras.metrics.BinaryAccuracy()])

        self.model.fit([X[:, i] for i in range(X.shape[1])],
                       y,
                       epochs=self.epochs)

    def predict(self, X=None):
        return (self.model.predict([X[:, i] for i in range(X.shape[1])])
                          .reshape(-1))
