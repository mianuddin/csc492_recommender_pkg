import numpy as np
from tensorflow import keras
from tensorflow.keras.layers.experimental \
    import preprocessing  # pylint: disable=no-name-in-module


def get_standard_layers(values, name=None):
    """Returns input layer and standard preprocessing layers for given values.

    Returns the input and preprocessing layers for the given integer values.
    The preprocessing consists of `IntegerLookup` and one-hot encoding via
    `CategoryEncoding`.

    Args:
        values (ndarray): The integer values of the desired input.
        name (String): The name of the values.

    Returns:
        Tuple[Layer, Layer]: The input and preprocessing layers.
    """
    unique_values = np.unique(values)
    input_layer = keras.Input(shape=(1), name=name, dtype="int64")
    indexer = preprocessing.IntegerLookup(max_tokens=len(unique_values))
    indexer.adapt(unique_values)
    encoder = preprocessing.CategoryEncoding(
        num_tokens=len(indexer.get_vocabulary()),
        output_mode="binary"
    )
    encoder.adapt(indexer(unique_values))
    pp_layers = encoder(indexer(input_layer))

    return input_layer, pp_layers
