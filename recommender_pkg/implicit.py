import numpy as np
import random
import tensorflow as tf
from sklearn.base import BaseEstimator
from tensorflow import keras


class ItemPopularity(BaseEstimator):
    """Recommender based solely on interactions per item."""

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
        unique, counts = np.unique(X[y == 1, 1], return_counts=True)
        self.interactions_by_item = dict(zip(unique, counts))

    def predict(self, X=None):
        """Predict the scores for the provided data.

        Args:
            X (ndarray of shape (n_samples, 2)): An array where each row
                                                 consists of a user and an
                                                 item.

        Returns:
            ndarray of shape (n_samples,): Class labels for each data sample.
        """
        y_pred = np.array([self.interactions_by_item[i] for i in X[:, 1]])
        return y_pred / max(y_pred)


class ImplicitKerasRecommender(BaseEstimator):
    """Abstract class for implicit recommenders built with Keras models."""
    def __init__(self,
                 epochs=10,
                 seed=None,
                 user_input=None,
                 item_input=None,
                 user_preprocessing_layers=None,
                 item_preprocessing_layers=None):
        # TODO: add optimizer, loss, metrics
        self.epochs = epochs
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

        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss=keras.losses.BinaryCrossentropy(),
                           metrics=[keras.metrics.BinaryAccuracy()])

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


class GeneralizedMatrixFactorization(ImplicitKerasRecommender):
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

    @staticmethod
    def create_core_layers(n_factors,
                           user_layers,
                           item_layers,
                           user_dense_kwdargs={},
                           item_dense_kwdargs={}):
        """Creates the core layers of the GMF model.

        Returns the hidden layers of the model. Specifically, the ones between
        the inputs and the visible, output layer.

        Args:
            n_factors (int): The number of latent factors
            user_layers (Layer): The input or preprocessing layers for the
                                 users.
            item_layers (Layer): The input or preprocessing layers for the
                                 items.
            user_dense_kwdargs (Dictionary): The keyword arguments for the
                                             user dense layer.
            item_dense_kwdargs (Dictionary): The keyword arguments for the
                                             item dense layer.

        Returns:
            Layer: The core layers of the model.
        """

        gmf_layers = [
            keras.layers.Dense(n_factors, **user_dense_kwdargs)(user_layers),
            keras.layers.Dense(n_factors, **item_dense_kwdargs)(item_layers)
        ]
        gmf_layers = keras.layers.Multiply()(gmf_layers)

        return gmf_layers

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

        gmf_layers = GeneralizedMatrixFactorization.create_core_layers(
            self.n_factors,
            user_preprocessing_layers,
            item_preprocessing_layers
        )

        gmf_output = keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_constraint=keras.constraints.unit_norm()
        )(gmf_layers)

        return keras.Model(inputs=[user_input, item_input],
                           outputs=[gmf_output],
                           name="generalized_matrix_factorization")

    def get_core_layers_kwdargs(self):
        """Returns the appropriate kwdargs for pretraining core layers.

            Returns:
                Dictionary, Dictionary: The keyword arguments for the user
                                        and item dense layers.
        """
        if not self.model:
            raise RuntimeError("GMF is not trained.")

        user_kernel, user_bias = self.model.layers[6].get_weights()
        item_kernel, item_bias = self.model.layers[7].get_weights()
        user_dense_kwdargs = {
            "kernel_initializer": keras.initializers.Constant(user_kernel),
            "bias_initializer": keras.initializers.Constant(user_bias)
        }
        item_dense_kwdargs = {
            "kernel_initializer": keras.initializers.Constant(item_kernel),
            "bias_initializer": keras.initializers.Constant(item_bias)
        }

        return user_dense_kwdargs, item_dense_kwdargs

    def get_output_weights(self):
        """Returns the kernel and bias for the output layer of this model.

            Returns:
                List[ndarray, Optional[ndarray]]: The kernel and bias.
        """
        if not self.model:
            raise RuntimeError("GMF is not trained.")

        return self.model.layers[-1].get_weights()


class MultiLayerPerceptron(ImplicitKerasRecommender):
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

    @staticmethod
    def create_core_layers(n_factors,
                           n_hidden_layers,
                           user_layers,
                           item_layers,
                           hidden_layers_kwdargs=[]):
        """Creates the core layers of the MLP model.

        Returns the hidden layers of the model. Specifically, the ones between
        the inputs and the visible, output layer.

        Args:
            n_factors (int): The number of latent factors
            user_layers (Layer): The input or preprocessing layers for the
                                 users.
            item_layers (Layer): The input or preprocessing layers for the
                                 items.
            hidden_layers_kwdargs (List[Dictionary, ...]): The keyword
                                                           arguments for each
                                                           hidden layer.

        Returns:
            Layer: The core layers of the model.
        """

        mlp_layers = keras.layers.Concatenate()([user_layers, item_layers])

        for x, i in enumerate(range(n_hidden_layers)[::-1]):
            current_kwdargs = {}

            if x < len(hidden_layers_kwdargs):
                current_kwdargs = hidden_layers_kwdargs[x]

            mlp_layers = keras.layers.Dense(n_factors * (2 ** i),
                                            activation="relu",
                                            **current_kwdargs)(mlp_layers)

        return mlp_layers

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

        mlp_layers = MultiLayerPerceptron.create_core_layers(
            self.n_factors,
            self.n_hidden_layers,
            user_preprocessing_layers,
            item_preprocessing_layers
        )

        mlp_output = keras.layers.Dense(1,
                                        activation="sigmoid",
                                        use_bias=False)(mlp_layers)

        return keras.Model(inputs=[user_input, item_input],
                           outputs=[mlp_output],
                           name="multi-layer_perceptron")

    def get_core_layers_kwdargs(self):
        """Returns the appropriate kwdargs for pretraining core layers.

            Returns:
                Dictionary: The keyword arguments for the hidden layers.
        """
        if not self.model:
            raise RuntimeError("MLP is not trained.")

        hidden_layers_kwdargs = []
        for i in range(7, 7 + self.n_hidden_layers):
            kernel, bias = self.model.layers[i].get_weights()
            hidden_layers_kwdargs.append({
                "kernel_initializer": keras.initializers.Constant(kernel),
                "bias_initializer": keras.initializers.Constant(bias)
            })

        return hidden_layers_kwdargs

    def get_output_weights(self):
        """Returns the kernel and bias for the output layer of this model.

            Returns:
                List[ndarray, Optional[ndarray]]: The kernel and bias.
        """
        if not self.model:
            raise RuntimeError("MLP is not trained.")

        return [self.model.layers[-1].get_weights()[0], None]


class NeuralMatrixFactorization(ImplicitKerasRecommender):
    """Recommender implementing the NeuMF architecture, an ensemble of GMF/MLP.

    Args:
        gmf_n_factors (int): The number of latent factors for GMF.
        mlp_n_factors (int): The number of latent factors for MLP.
        mlp_n_hidden_layers (int): The number of hidden layers.
        epochs (int): The number of epochs to train the NN.
        seed (int): A random seed.
        user_input (Keras.Input): An input for the users.
        item_input (Keras.Input): An input for the items.
        user_preprocessing_layers (Keras.Layer): Preprocessing layers for the
                                                 users.
        item_preprocessing_layers (Keras.Layer): Preprocessing layers for the
                                                 users.
        gmf_trained (GeneralizedMatrixFactorization): A trained GMF model of
                                                      the same number of
                                                      factors.
        mlp_trained (MultiLayerPerceptron): A trained MLP model of the same
                                            number of factors and hidden
                                            layers.
    """
    def __init__(self,
                 gmf_n_factors=8,
                 mlp_n_factors=8,
                 mlp_n_hidden_layers=4,
                 epochs=10,
                 seed=None,
                 user_input=None,
                 item_input=None,
                 user_preprocessing_layers=None,
                 item_preprocessing_layers=None,
                 gmf_trained=None,
                 mlp_trained=None,
                 alpha=0.5):
        self.gmf_n_factors = gmf_n_factors
        self.mlp_n_factors = mlp_n_factors
        self.mlp_n_hidden_layers = mlp_n_hidden_layers
        self.epochs = epochs
        self.seed = seed
        self.user_input = user_input
        self.item_input = item_input
        self.user_preprocessing_layers = user_preprocessing_layers
        self.item_preprocessing_layers = item_preprocessing_layers
        self.gmf_trained = gmf_trained
        self.mlp_trained = mlp_trained
        self.alpha = alpha

    def create_model(self):
        """Creates a new NeuMF model.

        Returns:
            Keras.Model: The NeuMF model. It will be pretrained if trained
                         models are provided in the constructor.
        """

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

        user_dense_kwdargs = {}
        item_dense_kwdargs = {}
        hidden_layers_kwdargs = []
        neumf_output_kernel = "glorot_uniform"

        if self.gmf_trained and self.mlp_trained:
            if self.gmf_trained.n_factors != self.gmf_n_factors:
                raise RuntimeError("GMF factors are not consistent.")

            if self.mlp_trained.n_factors != self.mlp_n_factors:
                raise RuntimeError("MLP factors are not consistent.")
            if self.mlp_trained.n_hidden_layers != self.mlp_n_hidden_layers:
                raise RuntimeError("MLP factors are not consistent.")

            user_dense_kwdargs, item_dense_kwdargs = (
                self.gmf_trained.get_core_layers_kwdargs()
            )

            hidden_layers_kwdargs = self.mlp_trained.get_core_layers_kwdargs()

            gmf_output_kernel, _ = self.gmf_trained.get_output_weights()
            mlp_output_kernel, _ = self.mlp_trained.get_output_weights()
            neumf_output_kernel = keras.initializers.Constant(
                np.concatenate((gmf_output_kernel * self.alpha,
                                mlp_output_kernel * (1 - self.alpha)))
            )

        gmf_layers = GeneralizedMatrixFactorization.create_core_layers(
            self.gmf_n_factors,
            user_preprocessing_layers,
            item_preprocessing_layers,
            user_dense_kwdargs,
            item_dense_kwdargs
        )

        mlp_layers = MultiLayerPerceptron.create_core_layers(
            self.mlp_n_factors,
            self.mlp_n_hidden_layers,
            user_preprocessing_layers,
            item_preprocessing_layers,
            hidden_layers_kwdargs
        )

        neumf_layers = [gmf_layers, mlp_layers]
        neumf_layers = keras.layers.Concatenate()(neumf_layers)
        neumf_layers = (
            keras.layers.Dense(1,
                               activation="sigmoid",
                               kernel_initializer=neumf_output_kernel,
                               kernel_constraint=keras.constraints.unit_norm(),
                               use_bias=False)(neumf_layers)
        )

        return keras.Model(inputs=[user_input, item_input],
                           outputs=[neumf_layers],
                           name="neural_matrix_factorization")
