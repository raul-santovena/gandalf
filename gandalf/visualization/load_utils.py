import glob
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import yaml
import pickle


def load_configuration_data(file_path):
    '''Load the file visualization_config.yaml as a dictionary'''

    # Load the configuration file
    _config_data = yaml.load(open(file_path, 'r'), Loader=yaml.Loader)

    # Obtaining the 'model' object from the json
    _config_dict = _config_data['model']

    return _config_dict


def load_scalers(file_path):
    '''Load sklearn scalers to reverse normalized data'''

    _config_dict = load_configuration_data(file_path)

    _base_dir = os.path.normpath(_config_dict['base_dir'])
    _model_id = _config_dict['model_id']
    _X_scaler_filename = _config_dict['X_scaler_filename'] if 'X_scaler_filename' in _config_dict else 'X_scaler.pkl'
    _y_scaler_filename = _config_dict['param_scaler_filename'] if 'param_scaler_filename' in _config_dict else 'param_scaler.pkl'

    _full_data_path = os.path.join(_base_dir, 'data', _model_id)

    # Load scalers (QuantileTransformers)
    _param_scaler = pickle.load(open(os.path.join(_full_data_path, _y_scaler_filename), 'rb'))
    _X_scaler = pickle.load(open(os.path.join(_full_data_path, _X_scaler_filename), 'rb'))

    return _X_scaler, _param_scaler


def load_data(file_path):
    '''Load the necessary data for the autoencoder'''

    _config_dict = load_configuration_data(file_path)

    # Building full data path
    _base_dir = os.path.normpath(_config_dict['base_dir'])
    _model_id = _config_dict['model_id']
    _full_data_path = os.path.join(_base_dir, 'data', _model_id)

    logging.info('Loading data from {}'.format(_full_data_path))

    # Filenames
    _X_filename = _config_dict['X_filename'] if 'X_filename' in _config_dict else 'X_test.npy'
    _params_filename = _config_dict['params_filename'] if 'params_filename' in _config_dict else 'params_test.npy'
    _cond_params_filename =_config_dict['cond_params_filename'] if 'cond_params_filename' in _config_dict else 'cond_params_test.npy'
    _z_filename = _config_dict['z_filename'] if 'z_filename' in _config_dict else 'z_test.npy'
    _decoded_filename = _config_dict['decoded_filename'] if 'decoded_filename' in _config_dict else 'decoded_test.npy'
    _ids_filename = _config_dict['ids_filename'] if 'ids_filename' in _config_dict else 'ids_test.npy'
    _axis_labels_filename = _config_dict['axis_labels_filename'] if 'axis_labels_filename' in _config_dict else 'axis_labels.npy'
    _params_names_filename = _config_dict['params_names_filename'] if 'params_names_filename' in _config_dict else 'params_names.npy'
    _cond_params_names_filename = _config_dict['cond_params_names_filename'] if 'cond_params_names_filename' in _config_dict else 'cond_params_names.npy'

    # Data
    _x = np.load(os.path.join(_full_data_path, _X_filename), allow_pickle=True)
    _z = np.load(os.path.join(_full_data_path, _z_filename), allow_pickle=True)
    _decoded = np.load(os.path.join(_full_data_path, _decoded_filename), allow_pickle=True)
    _ids = np.load(os.path.join(_full_data_path, _ids_filename), allow_pickle=True)
    _axis_labels = np.load(os.path.join(_full_data_path, _axis_labels_filename), allow_pickle=True)
    _cond_params = np.load(os.path.join(_full_data_path, _cond_params_filename), allow_pickle=True)
    _params = np.load(os.path.join(_full_data_path, _params_filename), allow_pickle=True)

    # Data limits
    _data_limits = _config_dict['data_limits'] if 'data_limits' in _config_dict else None
    if _data_limits is not None and _x.shape[0] > _data_limits:
        _x = _x[:_data_limits]
        _z = _z[:_data_limits]
        _decoded = _decoded[:_data_limits]
        _ids = _ids[:_data_limits]
        _cond_params = _cond_params[:_data_limits]
        _params = _params[:_data_limits]


    _df = pd.DataFrame.from_dict(data={'ids':_ids.tolist(), 'X':_x.tolist(),
                                 'z':_z.tolist(), 'decoded':_decoded.tolist(),
                                 })

    # Parameters names
    _params_names = np.load(os.path.join(_full_data_path, _params_names_filename), allow_pickle=True)
    _cond_params_names = np.load(os.path.join(_full_data_path, _cond_params_names_filename), allow_pickle=True)

    # We get each unscaled parameter and store them in an individual column
    _params_names = [_param_name.lower() for _param_name in _params_names]

    for _param_name in _params_names:
        _df['original_'+_param_name] = _params[:, np.isin(_params_names, [_param_name])].flatten().astype(str)

    # Idem with conditional parameters
    _cond_params_names = [_cond_param_name.lower() for _cond_param_name in _cond_params_names]

    for _cond_param_name in _cond_params_names:
        _df[_cond_param_name] = _cond_params[:, np.isin(_cond_params_names, [_cond_param_name])].flatten().astype(str)


    return _df, _axis_labels, _params_names, _cond_params_names


def load_model(file_path):
    '''Load the autoencoder model'''
    _config_dict = load_configuration_data(file_path)

    _base_dir = os.path.normpath(_config_dict['base_dir'])
    _model_id = _config_dict['model_id']
    _full_model_path = glob.glob(os.path.join(_base_dir, 'models', _model_id, 'autoencoder*'))[0]

    return tf.keras.models.load_model(filepath=_full_model_path)
