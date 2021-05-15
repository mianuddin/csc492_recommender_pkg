import math
import numpy as np
from tensorflow import keras
from tqdm import tqdm
from .recommenders import Recommender, KerasRecommender


class FunkSVD(Recommender):
    """Recommender implementing Funk SVD.

    Funk SVD without global baselines.

    Args:
        user (ndarray): An array of the users.
        item (ndarray): An array of the items.
        latent_factors (int): The number of latent factors.
        epochs (int): The number of epochs to train the NN.
        learning_rate (float): The learning rate of the model.
        regularization_term (float): The regularization term of the model.
        verbose (bool): Whether or not to print verbose output.
    """
    def __init__(self,
                 users,
                 items,
                 latent_factors=100,
                 epochs=10,
                 learning_rate=0.005,
                 regularization_term=0.02,
                 verbose=False):
        self.users = users
        self.items = items
        self.latent_factors = latent_factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization_term = regularization_term
        self.verbose = verbose

    def create_latent_factor_matrices(self):
        """Create matrices for the latent factors of the users and items.

        Creates the matrices which represent the factorization of the user-item
        matrix. In the user latent factor matrix, the rows are the users and
        the columns are the latent factors. In the item latent factor matrix,
        the rows are latent factors and the columns are the items.

        Returns:
            ndarray, ndarray: The latent factor matrices for users and items
                              respectively.
        """
        user_df = np.full((len(self.users), self.latent_factors), 0.1)
        item_df = np.full((self.latent_factors, len(self.items)), 0.1)
        return user_df, item_df

    def predict_rating(self, user_i, item_i):
        """Predict the rating for an item by the given user.

        Args:
            user_i (int): The user index.
            item_i (int): The item index.

        Returns:
            float: The predicted rating of the item by the user.
        """
        return np.dot(self.user_df[user_i], self.item_df[:, item_i])

    def process_users_items(self):
        """Create dictionaries mapping user and item ids to indexes.

        Replicates the functionality provided by Keras's `IntegerLookup`.
        """
        self.user_to_idx = {}
        self.idx_to_user = {}
        for i, u in enumerate(self.users):
            self.user_to_idx[u] = i
            self.idx_to_user[i] = u

        self.item_to_idx = {}
        self.idx_to_item = {}
        for i, x in enumerate(self.items):
            self.item_to_idx[x] = i
            self.idx_to_item[i] = x

    def train_pair(self, user, item, actual_rating):
        """Train the model on a single user-item pair.

        Args:
            user (int): The user id.
            item (int): The item id.
            actual_rating (float): The rating of the item by the user.

        Returns:
            float: The difference between the true and predicted ratings.
        """
        user_i = self.user_to_idx[user]
        item_i = self.item_to_idx[item]
        predicted_rating = self.predict_rating(user_i, item_i)
        error = actual_rating - predicted_rating

        for latent_factor in range(self.latent_factors):
            user_latent_factor = self.user_df[user_i, latent_factor]
            item_latent_factor = self.item_df[latent_factor, item_i]

            self.user_df[user_i, latent_factor] += (
                self.learning_rate *
                (error * item_latent_factor -
                 self.regularization_term * user_latent_factor))
            self.item_df[latent_factor, item_i] += (
                self.learning_rate *
                (error * user_latent_factor -
                 self.regularization_term * item_latent_factor))

        return error

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
        self.process_users_items()

        self.user_df, self.item_df = self.create_latent_factor_matrices()

        for epoch in (tqdm(range(self.epochs))
                      if self.verbose
                      else range(self.epochs)):
            errors = []

            for i in range(len(X)):
                user, item = X[i]
                rating = y[i]
                errors += [self.train_pair(user, item, rating)]

            if self.verbose:
                rmse = math.sqrt(sum([x ** 2 for x in errors]) / len(errors))

                if epoch == 0:
                    print("EPOCH\tRMSE")

                print(f"{epoch:02}\t{rmse:.8g}")

    def predict(self, X=None):
        """Predict the scores for the provided data.

        Args:
            X (ndarray of shape (n_samples, 2)): An array where each row
                                                 consists of a user and an
                                                 item.

        Returns:
            ndarray of shape (n_samples,): Class labels for each data sample.
        """
        return [self.predict_rating(self.user_to_idx[user],
                                    self.item_to_idx[item])
                for user, item in X]


class MatrixFactorization(KerasRecommender):
    """Recommender implementing Funk SVD with a NN.

    Args:
        n_factors (int): The number of latent factors.
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
                 n_factors=100,
                 epochs=10,
                 optimizer=keras.optimizers.SGD(),
                 loss=keras.losses.MeanSquaredError(),
                 metrics=[keras.metrics.RootMeanSquaredError(),
                          keras.metrics.MeanAbsoluteError()],
                 seed=None,
                 user_input=None,
                 item_input=None,
                 user_preprocessing_layers=None,
                 item_preprocessing_layers=None):
        super.__init__(epochs,
                       optimizer,
                       loss,
                       metrics,
                       seed,
                       user_input,
                       item_input,
                       user_preprocessing_layers,
                       item_preprocessing_layers)
        self.n_factors = n_factors
        self.user_input = user_input
        self.item_input = item_input
        self.user_preprocessing_layers = user_preprocessing_layers
        self.item_preprocessing_layers = item_preprocessing_layers

    @staticmethod
    def create_core_layers(n_factors,
                           user_layers,
                           item_layers):
        """Creates the core layers of the MF model.

        Returns the hidden layers of the model. Specifically, the ones between
        the inputs and the visible, output layer.

        Args:
            n_factors (int): The number of latent factors
            user_layers (keras.layers.Layer): The input or preprocessing layers
                                              for the users.
            item_layers (keras.layers.Layer): The input or preprocessing layers
                                              for the items.

        Returns:
            Layer: The core layers of the model.
        """

        mf_layers = [
            keras.layers.Dense(n_factors)(user_layers),
            keras.layers.Dense(n_factors)(item_layers)
        ]
        mf_layers = keras.layers.Multiply()(mf_layers)

        return mf_layers

    def create_model(self):
        """Creates a new MF model."""
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

        mf_layers = MatrixFactorization.create_core_layers(
            self.n_factors,
            user_preprocessing_layers,
            item_preprocessing_layers
        )

        mf_output = keras.layers.Dense(
            1,
            kernel_initializer=keras.initializers.Ones(),
            use_bias=False
        )(mf_layers)

        return keras.Model(inputs=[user_input, item_input],
                           outputs=[mf_output],
                           name="matrix_factorization")
