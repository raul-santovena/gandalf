import tensorflow as tf
import pickle
import glob
import os

def _load_scalers(base_dir, model_id):
    '''Load sklearn scalers to reverse normalized data'''
    FILE_PATH = os.path.abspath('')

    _full_data_path = os.path.join(FILE_PATH, base_dir, 'data', model_id)

    # Load scalers (QuantileTransformers)
    _param_scaler = pickle.load(open(os.path.join(_full_data_path, "param_scaler.pkl"), 'rb'))
    _X_scaler = pickle.load(open(os.path.join(_full_data_path, "X_scaler.pkl"), 'rb'))

    return _X_scaler, _param_scaler

def _load_model(base_dir, model_id):
    '''Load the autoencoder model'''
    FILE_PATH = os.path.abspath('')

    _full_model_path = glob.glob(os.path.join(FILE_PATH, base_dir, 'models', model_id, 'autoencoder*'))[0]

    return tf.keras.models.load_model(filepath=_full_model_path)

def generate_z(base_dir, model_id, data, parameters, input_without_params=False):
    """
    Generates reconstructed samples using a trained GANDALF model.

    Parameters
    ----------
    base_dir : str
        Base directory where the results are stored.
    model_id : str
        Identifier of the trained model.
    data : np.array, shape (N, M)
        Input data representing the spectra, where N is the number of samples and M is the length of one sample.
    parameters : np.array, shape (N, P)
        Original parameters corresponding to the input data, where P is the number of parameters.
    input_without_params : bool, optional
        If True, the encoder processes only `data` without additional parameters (default is False).

    Returns
    -------
    np.array, shape (N, L)
        Latent representations of the input data, where L is the dimensionality of the latent space.
    """

    # Scalers
    x_scaler, param_scaler = _load_scalers(base_dir, model_id)

    # Load model
    model = _load_model(base_dir, model_id)
    encoder = model.get_layer('encoder')

    # Param scaler
    parameters = param_scaler.transform(parameters)

    # Data (X) scaler
    data = data if x_scaler is None else x_scaler.transform(data)

    # To TensorFlow tensors
    data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)

    # Obtain the latent representation
    if input_without_params:
        z_tensor = encoder(data_tensor)
    else:
        params_tensor = tf.convert_to_tensor(parameters, dtype=tf.float32)
        z_tensor = encoder([data_tensor, params_tensor])

    return z_tensor.numpy()

def generate_samples_from_z(base_dir, model_id, z, target_parameter):
    """
    Generates reconstructed samples from latent representations using a trained GANDALF model.

    Parameters
    ----------
    base_dir : str
        Base directory where the model and scalers are stored.
    model_id : str
        Identifier of the trained model.
    z : np.array, shape (N, L)
        Latent representations of the input data, where N is the number of samples and L is the dimensionality of the latent space.
    target_parameter : np.array, shape (N, P)
        Target parameters for which the spectra should be reconstructed, where P is the number of parameters.

    Returns
    -------
    np.array, shape (N, M)
        Reconstructed spectra based on the target parameters, where M is the length of one sample.
    """

    # Scalers
    x_scaler, param_scaler = _load_scalers(base_dir, model_id)

    # Load model
    model = _load_model(base_dir, model_id)
    decoder = model.get_layer('decoder')

    # Param scaler
    target_parameter = param_scaler.transform(target_parameter)

    # To TensorFlow tensors
    z = tf.convert_to_tensor(z, dtype=tf.float32)
    target_parameter = tf.convert_to_tensor(target_parameter, dtype=tf.float32)

    # Reconstruct the spectrum
    reconstructed = decoder([z, target_parameter])

    # Convert to numpy array
    reconstructed = reconstructed.numpy()

    # Scaler if needed
    reconstructed = reconstructed if x_scaler is None else x_scaler.inverse_transform(reconstructed)

    return reconstructed

def generate_samples(base_dir, model_id, data, parameters, target_parameter, input_without_params=False):
    """
    Generates reconstructed samples using a trained GANDALF model.

    Parameters
    ----------
    base_dir : str
        Base directory where the results are stored.
    model_id : str
        Identifier of the trained model.
    data : np.array, shape (N, M)
        Input data representing the spectra, where N is the number of samples and M is the length of one sample.
    parameters : np.array, shape (N, P)
        Original parameters corresponding to the input data, where P is the number of parameters.
    target_parameter : np.array, shape (N, P)
        Target parameters for which the spectra should be reconstructed, having the same shape as `parameters`.
    input_without_params : bool, optional
        If True, the encoder processes only `data` without additional parameters (default is False).

    Returns
    -------
    np.array, shape (N, M)
        Reconstructed spectra based on the target parameters.
    """

    z = generate_z(base_dir, model_id, data, parameters, input_without_params=input_without_params)

    reconstructed = generate_samples_from_z(base_dir, model_id, z, target_parameter)

    return reconstructed