from keras.models import Model
from keras.layers import Flatten, Dense, Input
from collab.layers import get_embedding_layer, get_embedding_with_bias, element_wise_mul, concat, add_merge, dot_merge



def mlp_model(num_users, num_items, mlp_layers, embedding_dims=128, reg_embedding=0, merge_layer=concat, output_shape=1, output_activation='sigmoid'):
    """multi-layer perceptron model for collaberative filtering

    Args:
        num_users (int): number of users in dataset
        num_items (int): number of items in dataset
        mlp_layers (MLP object): the MLP part of the model
            see collab.blocks.MLP
        embedding_dims (int, optional): dimenstions of embedding space
        reg_embedding (float, optional): amount of l2 regularization penatly to apply 
            to embedding layers
        merge_layer (object, merge): merge function for embedding layers default (and reccomended)
            is 'concat'.
        output_shape (int, optional): model output shape
        output_activation (str, optional): defaults to 'sigmoid' for binary labels
            use linear for ratings prediction

    Returns:
        keras model object
    """
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    user_embedding = get_embedding_layer(
        num_users + 1, embedding_dims, reg=reg_embedding, name='user_embedding')(user_input)
    item_embedding = get_embedding_layer(
        num_items + 1, embedding_dims, reg=reg_embedding, name='item_embedding')(item_input)

    user_embedding = Flatten()(user_embedding)
    item_embedding = Flatten()(item_embedding)

    x = merge_layer([user_embedding, item_embedding])
    x = mlp_layers(x)
    out = Dense(output_shape, activation=output_activation, name="out")(x)
    model = Model(inputs=[user_input, item_input],
                  outputs=out)
    return model


def matrix_factorization_model(num_users, num_items, embedding_dims=128, reg_embedding=0,
                               merge_layer=dot_merge, output_shape=1, output_activation='sigmoid'):
    """matrix factorization model for collaberative filtering

    Args:
        num_users (int): number of users in dataset
        num_items (int): number of items in dataset
        mlp_layers (MLP object): the MLP part of the model
            see collab.blocks.MLP
        embedding_dims (int, optional): dimenstions of embedding space
        reg_embedding (float, optional): amount of l2 regularization penatly to apply 
            to embedding layers
        merge_layer (object, merge): merge function for embedding layers see collab.layers
            for ideas.
        output_shape (int, optional): model output shape
        output_activation (str, optional): defaults to 'sigmoid' for binary labels
            use linear for ratings prediction

    Returns:
        keras model object
    """

    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    user_embedding, user_bias = get_embedding_with_bias(num_users + 1, embedding_dims,
                                                        reg=reg_embedding, name='user_embedding')(user_input)

    item_embedding, item_bias = get_embedding_with_bias(num_items + 1, embedding_dims,
                                                        reg=reg_embedding, name='item_embedding')(item_input)

    x = merge_layer(user_embedding, item_embedding)
    x = add_merge(x, user_bias)
    x = add_merge(x, item_bias)

    out = Dense(output_shape, activation=output_activation, name="out")(x)
    model = Model(inputs=[user_input, item_input],
                  outputs=out)
    return model

