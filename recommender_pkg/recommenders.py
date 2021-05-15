import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from sklearn.base import BaseEstimator


class Recommender(BaseEstimator):
    """Abstract class for recommenders."""


class KerasRecommender(Recommender):
    """Abstract class for recommenders built with Keras models.
    
    Args:
        epochs (int): The number of epochs to train the NN.
        optimizer (keras.optimizers.Optimizer): The model's optimizer.
        loss (keras.losses.Loss): The loss function.
        metrics (List[keras.metrics.Metric, ...]): The metric functions.
        seed (int): A random seed.
        user_input (keras.Input): An input for the users.
        item_input (keras.Input): An input for the items.
        user_preprocessing_layers (keras.layers.Layer): Preprocessing layers
                                                        for the users.
        item_preprocessing_layers (keras.layers.Layer): Preprocessing layers
                                                        for the items.

    """
    def __init__(self,
                 epochs=10,
                 optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=[keras.metrics.BinaryAccuracy()],
                 seed=None,
                 user_input=None,
                 item_input=None,
                 user_preprocessing_layers=None,
                 item_preprocessing_layers=None):
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.seed = seed
        self.user_input = user_input
        self.item_input = item_input
        self.user_preprocessing_layers = user_preprocessing_layers
        self.item_preprocessing_layers = item_preprocessing_layers

    def create_model(self):
        """Creates a new Keras model."""
        pass

    def fit(self, X=None, y=None):
        """Fit the recommender from the training dataset.

        Args:
            X (ndarray of shape (n_samples, 2)): An array where each row
                                                 consists of a user and an
                                                 item.
            y (ndarray of shape (n_samples,)): An array where each entry
                                               denotes interactions between
                                               the corresponding user and item.
        """
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)

        # pylint: disable=assignment-from-no-return
        self.model = self.create_model()

        if not self.model:
            raise RuntimeError("Model was not created.")

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        self.history = self.model.fit([X[:, i] for i in range(X.shape[1])],
                                      y,
                                      epochs=self.epochs)

    def predict(self, X=None):
        """Predict the scores for the provided data.

        Args:
            X (ndarray of shape (n_samples, 2)): An array where each row
                                                 consists of a user and an
                                                 item.

        Returns:
            ndarray of shape (n_samples,): Class labels for each data sample.
        """
        return (self.model.predict([X[:, i] for i in range(X.shape[1])])
                          .reshape(-1))
