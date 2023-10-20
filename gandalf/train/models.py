import tensorflow as tf
from tensorflow.keras import layers

def make_encoder_model(data_dim, params_dim, latent_dim, hidden_layer_sizes=[512, 256], 
                       hidden_layers_activation='relu', output_activation='sigmoid',
                       input_only_spectra=False,
                       batch_norm=False, verbose=False):
    '''Create the encoder tensorflow model
    
    Parameters
    ----------
    data_dim : int
        Size of the input one-dimensional array that represents the data
        
    params_dim : int
        Size of the input one-dimensional array representing the parameters
        
    latent_dim : int
        Size of the output array, which will be the latent space of the autoencoder
        
    hidden_layer_sizes : list, length = n_layers - 2, default=[512, 256]
        The ith element represents the number of neurons in the ith hidden layer

    batch_norm : bool, default False
        If True, a normalization is added between each hidden layer

    verbose : bool, default True
    '''
    input_data = layers.Input(shape=(data_dim,))
    input_labels = layers.Input(shape=params_dim)

    if input_only_spectra:
        x = input_data
    else:
        x = layers.concatenate([input_data, input_labels])

    for hidden_layer_size in hidden_layer_sizes:
        x = layers.Dense(hidden_layer_size, activation=hidden_layers_activation)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
    x = layers.Dense(latent_dim, activation=output_activation)(x)
    
    model = tf.keras.Model(inputs=[input_data] if input_only_spectra else [input_data, input_labels], 
                           outputs=x,
                           name='encoder')
    
    verbose and print()
    verbose and model.summary()
    verbose and print()
       
    return model


def make_decoder_model(params_dim, latent_dim, data_dim, hidden_layer_sizes=[256, 512], 
                       hidden_layers_activation='relu', output_activation='sigmoid',
                       batch_norm=False, verbose=False):
    '''Create the tensorflow model of the decoder
    
    Parámetros
    ----------
    params_dim : int
        Size of the input one-dimensional array that represents the parameters
        
    latent_dim : int
        Size of the input array that represents the latent space of the autoencoder
        
    data_dim : int
        Size of the output one-dimensional array that represents the reconstruction of the data
        
    hidden_layer_sizes : list, length = n_layers - 2, default=[256, 512]
        The ith element represents the number of neurons in the ith hidden layer

    batch_norm : bool, default False
        If True, a normalization is added between each hidden layer

    verbose : bool, default True
    '''
    input_latent = layers.Input(shape=(latent_dim,))
    input_labels = layers.Input(shape=(params_dim,))
    
    x = layers.concatenate([input_latent, input_labels])
    
    for hidden_layer_size in hidden_layer_sizes:
        x = layers.Dense(hidden_layer_size, activation=hidden_layers_activation)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
    x = layers.Dense(data_dim, activation=output_activation)(x)
    
    model = tf.keras.Model(inputs=[input_latent, input_labels], outputs=x,
                           name='decoder')
    
    verbose and print()
    verbose and model.summary()
    verbose and print()

    return model


def make_autoencoder_model(data_dim, params_dim, latent_dim, encoder=None, decoder=None):
    '''Create the autoencoder tensorflow model
    
    Parameters
    ----------
    data_dim : int
        Size of the one-dimensional array that represents the data
        
    params_dim : int
        Size of the one-dimensional array that represents the parameters
        
    latent_dim : int
        Size of the array that represents the latent space of the autoencoder
        
    encoder : tf.keras.Model, default None
        Encoder tensorflow model, if not specified, one is created by default
        
    decoder : tf.keras.Model, default None
        Decoder tensorflow model, if not specified, one is created by default
    '''
    
    # If an encoder and/or a decoder is not passed, they are created with the default parameters
    if encoder is None:
        _encoder = make_encoder_model(data_dim=data_dim, params_dim=params_dim, latent_dim=latent_dim)
    else:
        _encoder = encoder
        
    if decoder is None:
        _decoder = make_decoder_model(data_dim=data_dim, params_dim=params_dim, latent_dim=latent_dim)
    else:
        _decoder = decoder
        
    input_data = layers.Input(shape=(data_dim,))
    input_labels = layers.Input(shape=(params_dim,))
    
    z = _encoder([input_data, input_labels])
    x = _decoder([z, input_labels])
    ae = tf.keras.Model(inputs=[input_data, input_labels], outputs=x, 
                           name='autoencoder')
    return ae


def make_discriminator(latent_dim, nbins, hidden_layer_sizes=[64, 32], 
                       convolutional=False, batch_norm=False,
                       output_activation='sigmoid', verbose=False):
    '''Create the tensorflow model of the discriminator
    
    Parameters
    ----------
    latent_dim : int
        Size of input one-dimensional array
        
    nbins : int
        Network output array size
        
    hidden_layer_sizes : list, length = n_layers - 2, default=[64, 32]
        The ith element represents the number of neurons in the ith hidden layer
        
    output_activation : {'sigmoid' o 'linear'}, default 'sigmoid'
        Last layer activation function

    verbose : bool, default True
    '''
    model = tf.keras.Sequential(name='discriminator')
    # If it is convolutional, an extra dimension is added
    if not convolutional:
        model.add(layers.InputLayer(input_shape=(latent_dim,)))
    else:
        model.add(layers.InputLayer(input_shape=(latent_dim, 1)))

    # Add as many hidden layers as indicated        
    for hidden_layer_size in hidden_layer_sizes:
        # The type of layer is added as indicated in the function parameters
        if not convolutional:
            model.add(layers.Dense(hidden_layer_size, activation='relu'))
        else:
            model.add(layers.Conv1D(filters=hidden_layer_size, kernel_size=3, padding='same'))
        # Normalization is added after each hidden layer if indicated
        if batch_norm:
            model.add(layers.BatchNormalization())
    # If it is a convolutional model, before producing the output, it is flattened and a normal layer is added prior to the output.
    if convolutional:
        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(nbins, activation=output_activation, dtype=tf.float64))

    verbose and print()
    verbose and model.summary()
    verbose and print()
    
    return model