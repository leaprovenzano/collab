from keras.layers import Embedding, Flatten
from keras.layers.merge import Multiply, Concatenate, Dot, Add
from keras.regularizers import l2

# --- EMBEDDING LAYERS ----#


def get_embedding_layer(input_dim, output_dim, reg=0., input_length=1, name=None):
    """constructor for embedding layer

    Args:
        input_dim (int): should be equal to number unique items in dataset
        output_dim (int): embedding dimensions
        reg (float, optional): amount of l2 regularization penatly to apply 
            to embedding.
        input_length (int, optional): defaults to 1 this is the length of the 
            input vector
        name (str, optional): highly reccommended to name the embedding
            layer especially in the case where there are multiple embeddings 
            in a model

    """
    return Embedding(input_dim, output_dim=output_dim,
                     embeddings_regularizer=l2(reg), input_length=input_length,
                     name=name)


def bias_layer(input_dim, input_length=1, name=None):
    """ a bias layer to be added back in to the input after embedding layer in 
    matrix factorization models"""
    return Embedding(input_dim, 1, input_length=input_length, name=name)


def get_embedding_with_bias(input_dim, output_dim, name, reg=0., input_length=1):
    """ get embedding and bias layers together given args to `get_embedding_layer`"""
    def make_layers(inp):
        emb = get_embedding_layer(
            input_dim, output_dim, reg, name=name, input_length=input_length)(inp)
        bias = bias_layer(input_dim, input_length=input_length,
                          name='{}_bias'.format(name))(inp)
        bias = Flatten()(bias)
        return emb, bias
    return make_layers


# --- MERGE LAYERS ----#

def add_merge(e1, e2):
    return Add()([e1, e2])


def element_wise_mul(e1, e2):
    return Multiply()([Flatten()(e1), Flatten()(e2)])


def dot_merge(e1, e2, axes=-1):
    return Flatten()(Dot(axes=axes)([e1, e2]))


def cos_merge(e1, e2, axes=-1):
    return Flatten()(Dot(axes=axes, normalize=True)([e1, e2]))


def concat(*args):
    return Concatenate()(*args)
