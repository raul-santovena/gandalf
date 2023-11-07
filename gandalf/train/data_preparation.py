#!/usr/bin/env python
# coding: utf-8

import os
import sys
from typing import Type
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder, KBinsDiscretizer, Normalizer
import tensorflow as tf
from abc import ABC, abstractmethod

# Abstract Class
class DataLoader(ABC):

    @abstractmethod
    def get_dataset_dir(self):
        raise NotImplementedError("Method not implemented!")
    
    @abstractmethod
    def get_dataset_name(self):
        raise NotImplementedError("Method not implemented!")
    
    @abstractmethod
    def get_name_labels(self):
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    def get_original_params_train(self):
        raise NotImplementedError("Method not implemented!")
        
    @abstractmethod
    def get_original_params_test(self):
        raise NotImplementedError("Method not implemented!")
        
    @abstractmethod
    def get_ids_train(self):
        raise NotImplementedError("Method not implemented!")
        
    @abstractmethod
    def get_ids_test(self):
        raise NotImplementedError("Method not implemented!")
        
    @abstractmethod
    def get_axis_labels(self):
        raise NotImplementedError("Method not implemented!")
        
    @abstractmethod
    def get_scaler_X(self):
        raise NotImplementedError("Method not implemented!")
        
    @abstractmethod
    def get_scaler_params(self):
        raise NotImplementedError("Method not implemented!")
        
    @abstractmethod
    def create_tf_dataset(self, batch=None, multi_discriminator=False, verbose=0) -> (tf.data.Dataset, tf.data.Dataset):
        raise NotImplementedError("Method not implemented!")

# Data loader Factory
class DataLoaderFactory:
    def __init__(self) -> None:
        pass

    def create_dataloader(self, data_loader, dataset_path, params, conditional_params,
                    normalize=True, shuffle=True, discretize=True, nbins=10, 
                    random_state=None, verbose=0) -> Type[DataLoader]:
        ''' Return a DataLoader based on the data_loader parameter using dynamic instantiation'''
        try:
            dataLoader = globals()[data_loader]
            verbose and print("Using the '{:}' class to prepare the datasets.".format(data_loader))
            return dataLoader(data_path=dataset_path, normalize_X=normalize, shuffle=shuffle, 
                              discretize=discretize, nbins=nbins, params=params, 
                              conditioned_params=conditional_params,
                              random_state=random_state)
        except:
            verbose and print("The '{:}' class does not exist in data_preparation.py. Using the default class 'SampleDataLoader'.".format(data_loader))
            return SampleDataLoader(data_path=dataset_path, normalize_X=normalize, shuffle=shuffle, discretize=discretize,
                                    nbins=nbins, params=params, conditioned_params=conditional_params,
                                    random_state=random_state)

# # Sample dataset

class SampleDataLoader(DataLoader):
    
    def __init__(self, dataset_name=None,
                 data_path=None,
                 normalize_X=False, shuffle=True,
                 discretize=True, nbins=10,
                 params=['teff', 'logg', 'metal', 'alpha'],
                 conditioned_params=['teff'],
                 random_state=1):
        
        # Sample data path
        DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/sample')

        # Sample files
        X_FILENAME = "X.npy"
        PARAMS_FILENAME = "params.npy"
        AXIS_LABELS_FILENAME = "axis_labels.npy"
        IDS_FILENAME = "ids.npy"


        _data_path = DATA_PATH if data_path is None else data_path
        _data_path = os.path.normpath(_data_path)

        self.dataset_dir = os.path.dirname(_data_path)
        self.dataset_name = os.path.basename(_data_path) if dataset_name is None else dataset_name

        _X_path = os.path.join(_data_path, X_FILENAME)
        _params_path = os.path.join(_data_path, PARAMS_FILENAME)
        _axis_labels_path = os.path.join(_data_path, AXIS_LABELS_FILENAME)
        _ids_path = os.path.join(_data_path, IDS_FILENAME)

        self.random_state = random_state
        self.normalize_X = normalize_X
        self.shuffle = shuffle
        self.discretize = discretize
        self.nbins = nbins
        self.name_labels = [p.lower() for p in params] # save and transform to lowercase
        
        self.original_X = self.__load_X(path=_X_path)
        self.original_params = self.__load_params(path=_params_path)
        self.conditioned_params = self.__load_conditioned_params(conditioned_params=[p.lower() for p in conditioned_params])
        self.ids = self.__load_ids(path=_ids_path)
        self.axis_labels = self.__load_axis_labels(path=_axis_labels_path)
        
        self.qt_params = QuantileTransformer(random_state=self.random_state)
        self.qt_X = QuantileTransformer(random_state=self.random_state)

    def get_dataset_dir(self):
        return self.dataset_dir
        
    def get_dataset_name(self):
        return self.dataset_name

    def get_name_labels(self):
        return self.name_labels
    
    def get_original_X(self):
        return self.original_X
    
    def get_original_params_df(self):
        return self.original_params_df
    
    def get_original_params(self):
        return self.original_params
    
    def get_original_params_train(self):
        return self.original_params_train
    
    def get_original_params_test(self):
        return self.original_params_test
    
    def get_conditioned_params(self):
        return self.conditioned_params
    
    def get_ids(self):
        return self.ids
    
    def get_ids_train(self):
        return self.ids_train
    
    def get_ids_test(self):
        return self.ids_test
    
    def get_axis_labels(self):
        return self.axis_labels
    
    def get_scaler_params(self):
        return self.qt_params
    
    def get_scaler_X(self):
        return self.qt_X
    
    def __load_axis_labels(self, path):
        _axis_labels = np.load(path, allow_pickle=True)

        return _axis_labels

    def __load_X(self, path, verbose=0):
        _X = np.load(path, allow_pickle=True)
        
        # Replace nans por ceros
        np.nan_to_num(_X, copy=False)

        verbose and print(_X.shape)

        return _X

    def __load_params(self, path):
        _params = np.load(path, allow_pickle=True)

        # Replace nans por ceros
        np.nan_to_num(_params, copy=False)
        
        self.original_params_df = pd.DataFrame(data=_params,
                                               columns=self.name_labels)
        
        return _params

    def __load_conditioned_params(self, conditioned_params): 
        # IMPORTANT! It is necessary to call __load_params first, because it initializes the 'original_params_df' attribute
        return self.original_params_df[conditioned_params].values.astype(np.float64)
    
    def __load_ids(self, path):
        _ids = np.load(path)
        return _ids
    
    def __discretize_and_encode_params(self, param_array, nbins=10, multi_discriminator=False):
        '''Discretize each parameter column into n bins, encode each unique combination into a string, and convert
        the final array in a one-hot vector'''
        # Discretize columns into n bins
        _kbd = KBinsDiscretizer(n_bins=nbins, strategy='uniform', encode='ordinal')

        _discretized_params = _kbd.fit_transform(param_array)

        if not multi_discriminator:
            # Create an encoding for each unique parameter combination (previously discretized) Example: [[0,0,7],[8,4,9]] = [['007'],['849']]
            _encoded_params = list()
            for params in _discretized_params.astype(int):
                _row = str()
                # A string is created that merges each discretized column
                for param in params:
                    _row += str(param)
                _encoded_params.append([_row])
            _encoded_params = np.array(_encoded_params)

            # OneHotEncoder
            _ohe = OneHotEncoder()

            return [_ohe.fit_transform(_encoded_params).toarray()]
        else:
            _encoded_params = list()
            for i in range(_discretized_params.shape[1]):
                _ohe = OneHotEncoder()
                _encoded_param = _ohe.fit_transform(_discretized_params[:,i].reshape(-1,1)).toarray()
                _encoded_params.append(_encoded_param)

            return _encoded_params
            
    def create_tf_dataset(self, batch=None, multi_discriminator=False, verbose=0):
        
        # Check nans
        assert not np.any(np.isnan(self.original_X))
        assert not np.any(np.isnan(self.conditioned_params))
        
        # Normalization
        _params_normalized = self.qt_params.fit_transform(self.conditioned_params)
        _x = self.qt_X.fit_transform(self.original_X) if self.normalize_X else self.original_X
            
        _y = _params_normalized
            
        # Discretization and One-hot encoding
        _y_encoded_list = self.__discretize_and_encode_params(self.conditioned_params, nbins=self.nbins,
                                                              multi_discriminator=multi_discriminator)

        # Train/Test split
        _X_train, _X_test, _y_train, _y_test, _original_params_train, _original_params_test, _ids_train, _ids_test, *_y_train_test_encoded_list = train_test_split(_x, 
                                                                                                                          _y,  
                                                                                                                          self.original_params, 
                                                                                                                          self.ids,
                                                                                                                          *_y_encoded_list,
                                                                                                                          random_state=self.random_state, 
                                                                                                                          shuffle=self.shuffle)
        # The split of ids and parameters are saved as class attributes
        self.original_params_train = _original_params_train
        self.original_params_test = _original_params_test
        self.ids_train = _ids_train
        self.ids_test = _ids_test
        
        verbose and print(_X_train.shape)
        verbose and print(_X_test.shape)
        verbose and print(_y_train.shape)
        verbose and print(_y_test.shape)
        if verbose:
            for _y in _y_train_test_encoded_list:
                print(_y.shape)
        
        _y_train_encoded_list = _y_train_test_encoded_list[::2] # elementos pares de la lista (train)
        _y_test_encoded_list = _y_train_test_encoded_list[1::2] # elementos impares de la lista (test)
        _train_dataset = tf.data.Dataset.from_tensor_slices((_X_train, _y_train, *_y_train_encoded_list))
        _test_dataset = tf.data.Dataset.from_tensor_slices((_X_test, _y_test, *_y_test_encoded_list))
        
        if batch:
            _train_dataset = _train_dataset.batch(batch)
            _test_dataset = _test_dataset.batch(batch)
            
        return _train_dataset, _test_dataset
# ---