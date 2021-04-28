from tensorflow import keras
from tensorflow.keras.layers.experimental \
    import preprocessing  # pylint: disable=no-name-in-module


def get_standard_layers(values, name=None):
    """Returns input layer and standard preprocessing layers for given values.
    """
    input_layer = keras.Input(shape=(1), name=name, dtype="int64")
    indexer = preprocessing.IntegerLookup(max_tokens=len(values))
    indexer.adapt(values)
    encoder = preprocessing.CategoryEncoding(
        num_tokens=len(indexer.get_vocabulary()),
        output_mode="binary"
    )
    encoder.adapt(indexer(values))
    pp_layers = encoder(indexer(input_layer))

    return input_layer, pp_layers
