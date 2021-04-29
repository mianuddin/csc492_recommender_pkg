import numpy as np
import random
import tensorflow as tf
from sklearn.base import ClassifierMixin
from tensorflow import keras


class GeneralizedMatrixFactorization(ClassifierMixin):
    """Recommender implementing the GMF architecture.

    Args:
        n_factors: The number of latent factors
        epochs: The number of epochs to train the NN
        seed: A random seed
        user_input: A Keras input for the users
        item_input: A Keras input for the items
        user_preprocessing_layers: Keras preprocessing layers for the users
        item_preprocessing_layers: Keras preprocessing layers for the items
    """
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
        """Creates a new GMF model."""
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
        """Fit the classifier from the training dataset.

        Args:
            X: ndarray of shape (n_samples, 2)
            y: ndarray of shape (n_samples,)
        """
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)

        self.model = self.create_model()
        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss=keras.losses.BinaryCrossentropy(),
                           metrics=[keras.metrics.BinaryAccuracy()])

        self.history = self.model.fit([X[:, i] for i in range(X.shape[1])],
                                      y,
                                      epochs=self.epochs)

    def predict(self, X=None):
        """Predict the class labels for the provided data.

        Args:
            X: ndarray of shape (n_samples, 2)

        Returns:
            y: Class labels for each data sample.
        """
        return (self.model.predict([X[:, i] for i in range(X.shape[1])])
                          .reshape(-1))


class MultiLayerPerceptron(ClassifierMixin):
    """Recommender implementing the MLP architecture.

    Args:
        n_factors: The number of latent factors
        n_hidden_layers: The number of hidden layers
        epochs: The number of epochs to train the NN
        seed: A random seed
        user_input: A Keras input for the users
        item_input: A Keras input for the items
        user_preprocessing_layers: Keras preprocessing layers for the users
        item_preprocessing_layers: Keras preprocessing layers for the items
    """
    def __init__(self,
                 n_factors=8,
                 n_hidden_layers=4,
                 epochs=10,
                 seed=None,
                 user_input=None,
                 item_input=None,
                 user_preprocessing_layers=None,
                 item_preprocessing_layers=None):
        self.n_factors = n_factors
        self.n_hidden_layers = n_hidden_layers
        self.epochs = epochs
        self.seed = seed
        self.user_input = user_input
        self.item_input = item_input
        self.user_preprocessing_layers = user_preprocessing_layers
        self.item_preprocessing_layers = item_preprocessing_layers

    def create_model(self):
        """Creates a new MLP model."""

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

        mlp_layers = keras.layers.Concatenate()([user_preprocessing_layers,
                                                 item_preprocessing_layers])

        for i in range(self.n_hidden_layers)[::-1]:
            mlp_layers = keras.layers.Dense(self.n_factors * (2 ** i),
                                            activation="relu")(mlp_layers)

        mlp_layers = keras.layers.Dense(1,
                                        activation="sigmoid",
                                        use_bias=False)(mlp_layers)

        return keras.Model(inputs=[user_input, item_input],
                           outputs=[mlp_layers],
                           name="multi-layer_perceptron")

    def fit(self, X=None, y=None):
        """Fit the classifier from the training dataset.

        Args:
            X: ndarray of shape (n_samples, 2)
            y: ndarray of shape (n_samples,)
        """
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)

        self.model = self.create_model()
        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss=keras.losses.BinaryCrossentropy(),
                           metrics=[keras.metrics.BinaryAccuracy()])

        self.history = self.model.fit([X[:, i] for i in range(X.shape[1])],
                                      y,
                                      epochs=self.epochs)

    def predict(self, X=None):
        """Predict the class labels for the provided data.

        Args:
            X: ndarray of shape (n_samples, 2)

        Returns:
            y: Class labels for each data sample.
        """
        return (self.model.predict([X[:, i] for i in range(X.shape[1])])
                          .reshape(-1))
