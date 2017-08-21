from keras import backend as K

from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Dropout, AlphaDropout, BatchNormalization, Embedding
from keras.regularizers import l2


class Block(object):
    """building block base object

    Attributes:

        batch_norm (bool): if true place a batch normalization layer before each activation layer. 
            *Note:* when selecting 'selu' as an activation layer this will be automatically overwritten 
            as false and no batch norm will be applied. see:
            [Self-Normalizing Neural Networks - Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)

        dropout (Keras.layer): again this is a special check for the SeLU activation which 
            requires AlphaDropout, otherwise this will be a standard keras Dropout layer object 

        kernal_init (str | keras.initilizer): initilizer for layers, defaults to glorot uniform (aka Xavier uniform)
            unless a SeLU activation is used, in which case defaults to lecun normal
    """

    # check backend for batch normalization axis.
    channel_axis = -1 if K.image_dim_ordering() is 'tf' else 1

    def __init__(self, activation='relu', batch_norm=True):
        self._activation = activation
        self.batch_norm = batch_norm and (activation is not 'selu')
        self.dropout = AlphaDropout if activation is 'selu' else Dropout
        self.kernal_init = 'lecun_normal' if activation is 'selu' else 'glorot_uniform'

    def activation(self):
        """hacky check for advanced activations.
        turn strings and activation functions into 
        activation functions without making a fuss.
        """
        if type(self._activation) is not str:
            return self._activation()
        return Activation(self._activation)


class MLP(Block):
    """
    shortcut multilayer perceptron builder. call like a regular
    keras layer:
    ie : x = MLP(**args)(inp) 

    Attributes:

        dropout_growth_factor (float): multiply dropout rate by
            dropout_growth_factor at each layer, if 0 no dropout will be applied.

        initial_drop (float): initial_dropout rate.

        initial_units (int): number of nodes in the first layer.

        input_drop (float): drop_out rate to be applied to first
            input layer may be if 0 no dropout will be applied.

        kwargs : additional keyword arguments to be passed to keras.layers.Dense 
            constructor. 
            *Note* that the MLP  will call pass the following args to 
            keras.layers.Dense - nodes, kernel_initilizer

        max_drop (float): maximum level of dropout, after reaching this point
            dropout will stop increasing.

        n_layers (int): number of layers to stack where a layer generally consists
            of dense layer > optional batch norm > activation > optional dropout

        unit_growth_factor (float): increase or decrease the number of units in each
            layer by this factor. for best results keep this semi-insync with 
            dropout_growth_factor.
    """


    def __init__(self, n_layers=1, activation='relu', batch_norm=True, initial_units='use_input',
                 initial_drop=0.1, unit_growth_factor=1., dropout_growth_factor=1.,
                 max_drop=.5, input_drop=0., **kwargs):
        super().__init__(activation, batch_norm)
        self.initial_units = initial_units
        self.n_layers = n_layers
        self.unit_growth_factor = unit_growth_factor
        self.dropout_growth_factor = dropout_growth_factor
        self.initial_drop = initial_drop
        self.max_drop = max_drop
        self.input_drop = input_drop
        self.kwargs = kwargs

    def __call__(self, inp):
        return self.build()(inp)

    def get_input_nodes(self, inp):
        """get the number of nodes from the input layer"""
        return inp.shape.as_list()[-1]

    def dense_layer(self, units):
        """only units need be specified kwargs set at init will be
        passed to keras.layers.Dense constructor
        """
        def block(inp):
            x = Dense(units, kernel_initializer=self.kernal_init,
                      **self.kwargs)(inp)
            if self.batch_norm:
                x = BatchNormalization(scale=False, axis=-1)(x)
            x = self.activation()(x)
            return x
        return block

    def build(self):
        """build function does the actual block building work
        and returns enclosed block.
        """
        def block(inp):
            if self.initial_units is 'use_input':
                self.initial_units = self.get_input_nodes(inp)

            # input dropout if any
            x = self.dropout(self.input_drop)(inp)

            units = self.initial_units
            drop_p = self.initial_drop

            for i in range(self.n_layers):

                x = self.dense_layer(int(units))(x)
                x = self.dropout(drop_p)(x)
                # grow or decrease units dense units and dropout
                units *= self.unit_growth_factor

                drop_p *= min((self.max_drop, drop_p *
                               self.dropout_growth_factor))
            return x
        return block
