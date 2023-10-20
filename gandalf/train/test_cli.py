#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer

from bokeh.models import ColorBar, ColumnDataSource
from bokeh.plotting import figure, show
from bokeh.transform import linear_cmap
from bokeh.palettes import RdYlBu9
from bokeh.layouts import row, column
from bokeh.io import export_png

import argparse

sys.path.append('../../gandalf') # To support the use of the tool without packaging
import gandalf.train.utils as my_utils

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# # Internal constants
# ## 
ROOT_FOLDER = None

# ## Tables
WIDTH = 100
LATEX_TABLE_TEMPLATE = "\\begin{{table}}[] \n" \
                       "\centering\n" \
                       "\\begin{{tabular}}{{@{{}}{:}@{{}}}}\n" \
                       "\\toprule\n" \
                       "    & \multicolumn{{{:}}}{{c}}{{\\(R^2\\) score}} \\\\ \midrule\n" \
                       "    {:}\\\\\n" \
                       "{:} \\bottomrule\n" \
                       "\end{{tabular}}\n" \
                       "\caption{{Caption}}\n" \
                       "\label{{tab:my-table}}\n" \
                       "\end{{table}}\n"

ALINGMENT = "l"

LATEX_HEADER_TEMPLATE = "& {:}"

LATEX_ROW_TEMPLATE = "{:} {:} \\\\\n"

LATEX_CELL_TEMPLATE = " & {:.4f}"



def _tsne_bokeh(embedded_x, embedded_z, y_cond, cond_name_labels, id, name=None, export_graph=False, verbose=1):
    verbose and print('Generating graph... ', end='')
    
    _colors = RdYlBu9[::-1]

    rows = list()

    for i, cond_name_label in enumerate(cond_name_labels):
        _y = y_cond[:, i]

        source = ColumnDataSource(dict(z_x=embedded_z[:,0],
                                    z_y=embedded_z[:,1],
                                    x_x=embedded_x[:,0],
                                    x_y=embedded_x[:,1],
                                    cond_label=_y))

        # use the field name of the column source
        cmap = linear_cmap(field_name='cond_label', palette=_colors, low=min(_y), high=max(_y))


        p1 = figure(width=900, height=900, title="")
        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.axis.visible = False
        p1.scatter(x='x_x', y='x_y', color=cmap, size=15, source=source)

        p2 = figure(width=900, height=900, title="")
        p2.toolbar.logo = None
        p2.toolbar_location = None
        p2.axis.visible = False
        p2.scatter(x='z_x', y='z_y', color=cmap, size=15, source=source)

        # pass the mapper's transform to the colorbar
        color_bar = ColorBar(color_mapper=cmap['transform'], width=10, title=cond_name_label)

        p1.add_layout(color_bar, 'right')
        p2.add_layout(color_bar, 'right')

        rows.append(row(p1,p2))

    full_layout = column(*rows)
    show(full_layout)

    if export_graph:
        filename = 'tsne_comparison_{:}_{:}.png'.format(id, '_'.join(cond_name_labels))
        filename_path = os.path.join(ROOT_FOLDER, filename)
        export_png(full_layout, filename=filename_path)
        verbose and print('and saving it at {:}'.format(os.path.abspath(filename_path)), end='\n\n')
    else:
        verbose and print('Done', end='\n\n')


def tsne_comparison(id_, name=None, random_state=1, cache='tests/tsne',
                    export_graph=False,
                    verbose=0):
    '''Compares the distribution of the original data and the latent space of the autoencoder, based on the 
    parameters, and using the tsne algorithm.
    
    Parameters
    ----------
    id_ : str
        Identifier from which the trained model and the data corresponding to it are obtained.
    name: str, default None
        Name of the model to put as the title of the graph, if not specified, the id_ is used
    random_state: int, default 1
        Seed for the tsne algorithm, in order to be able to replicate the results
    cache: str, default 'tests/tsne'
        If it is not None, look in the indicated path for the folder named "id_", the files "x_<random_state>.npy" and 
        "z_<random_state>.npy". If they exist, it loads them instead of running tsne. If it does not exist, run tsne and
        save the results.
    export_graph: bool, default False
        Export the graph to disk in png format
    verbose: int, default 0
        Verbose
    '''
    
    print('-'*WIDTH)
    print('t-SNE comparison'.center(WIDTH))
    print('-'*WIDTH)
    
    verbose and print('Starting tsne comparison...')
    # Load data
    data_dict = my_utils.load_data(id_=id_, base_dir=ROOT_FOLDER)
    
    # It checks if there is saved data related to the application of the tsne to this dataset with the same random_state
    saved_embedded_x_path = os.path.normpath(os.path.join(ROOT_FOLDER, cache, id_, 'x_{}.npy'.format(random_state)))
    saved_embedded_z_path = os.path.normpath(os.path.join(ROOT_FOLDER, cache, id_, 'z_{}.npy'.format(random_state)))
    
    ## X
    if not os.path.exists(saved_embedded_x_path): # If there is no previous data, the tsne is executed and the data is saved
        verbose and print('There is no cached data. Applying tsne to the original spectra... ', end='')
        tsne = TSNE(n_components=2, n_iter=1000, perplexity=30, learning_rate='auto', init='pca',
                    random_state=random_state, n_jobs=-1, verbose=0)

        embedded_x = tsne.fit_transform(data_dict['X'])
        
        # Create folder if doesn't exist
        _folder_path = os.path.dirname(saved_embedded_x_path)
        if not os.path.exists(_folder_path):
            print('Folder \'{:}\' doesn\'t exist! Created.'.format(os.path.abspath(_folder_path)))
            os.makedirs(_folder_path)
            
        np.save(saved_embedded_x_path, embedded_x)
        verbose and print('Done. Data saved in {}'.format(os.path.abspath(saved_embedded_x_path)))
        
    else: # If previous data exists, it is loaded
        verbose and print('Located data saved in {}'.format(os.path.abspath(saved_embedded_x_path)))
        embedded_x = np.load(saved_embedded_x_path, allow_pickle=True)
        
    ## z
    if not os.path.exists(saved_embedded_z_path): # If there is no previous data, the tsne is executed and the data is saved
        verbose and print('There is no cached data. Applying tsne to the latent space... ', end='')
        tsne = TSNE(n_components=2, n_iter=1000, perplexity=30, learning_rate='auto', init='pca',
                    random_state=random_state, n_jobs=-1, verbose=0)
        
        embedded_z = tsne.fit_transform(data_dict['z'])
        
        # Create folder if doesn't exist
        _folder_path = os.path.dirname(saved_embedded_z_path)
        if not os.path.exists(_folder_path):
            print('Folder \'{:}\' doesn\'t exist! Created.'.format(os.path.abspath(_folder_path)))
            os.makedirs(_folder_path)
        
        np.save(saved_embedded_z_path, embedded_z)
        verbose and print('Done. Data saved in {}'.format(os.path.abspath(saved_embedded_z_path)))
    else: # If previous data exists, it is loaded
        verbose and print('Located data saved in {}'.format(os.path.abspath(saved_embedded_z_path)))
        embedded_z = np.load(saved_embedded_z_path, allow_pickle=True)
    # -----------------------------------------------------------------------------------------------------------------------------
        
        
    # Plot
    _tsne_bokeh(embedded_x=embedded_x, embedded_z=embedded_z, 
                y_cond=data_dict['cond_params'], 
                cond_name_labels=data_dict['cond_params_names'],
                id=id_,
                name=name if name is not None else id_,
                export_graph=export_graph,
                verbose=1)
        

def _initialize_results_dict(ids, name_labels):
    _results_dict = dict()
    for _label in name_labels:
        # The dictionary is initialized
        _results_dict[_label] = dict()
        # Score lists are initialized
        _results_dict[_label]['regr_X_score'] = list()
        for _id in ids:
            _results_dict[_label][_id+'_regr_z_score'] = list()

    return _results_dict

def _print_plain_table(ids, results_dict, keys):
        '''Displays the results saved in the dictionary, which has the following
        structure:
            {'label_1':{'regr_X_score': <float_value>,
                        '{id_0}_regr_z_score': <float_value>,
                        '{id_1}_regr_z_score': <float_value>},
             .
             .
             .
             
             'label_n':{'regr_X_score': <float_value>,
                        '{id_0}_regr_z_score': <float_value>,
                        '{id_1}_regr_z_score': <float_value>}
            }
        
        Parameters
        ----------
        results_dict : dict
            Results dictionary
        keys: str list
            Primary keys of the dictionary, which represent the different 
            parameters being predicted
        '''
        _left_margin = ' ' * 10
        _division_line = '-' * ((10+10*len(ids)+1)+len(ids)) # left_margin + 10*(z_regressors + X_regressor) + n_separators 

        print()
        # First line
        print('{} {}'.format(_left_margin, 'r2 score'.center(20))) 
        # Division line
        print('{} {}'.format(_left_margin, _division_line))
        # Second line
        print('{} {}'.format(_left_margin,'regr_X'.center(10)), end='')
        for _i in range(len(ids)):
            print('|{}'.format('regr_z_{}'.format(_i+1).center(10)), end='')
        print()
        # Division line
        print('{} {}'.format(_left_margin, _division_line))
        # As many rows as conditional parameters
        for key in keys:
            print('{}|{:10.4f}'.format(key.center(10), results_dict[key]['regr_X_score']), 
                  end='')
            for _id in ids:
                print('{:10.4f}'.format(results_dict[key][_id+'_regr_z_score']),
                      end='')
            print()
        print()

def _print_latex_table(ids, results_dict, keys):
    _alingment = ALINGMENT*(len(ids)+2)

    _header = LATEX_HEADER_TEMPLATE.format('regr\\_X')
    for _i, _id in enumerate(ids):
        _header += LATEX_HEADER_TEMPLATE.format('regr\\_z\\_{}'.format(_i+1))

    _rows = ""
    for key in keys:
        _row = LATEX_CELL_TEMPLATE.format(results_dict[key]['regr_X_score'])
        for _id in ids:
            _row += LATEX_CELL_TEMPLATE.format(results_dict[key][_id+'_regr_z_score'])
        _rows += LATEX_ROW_TEMPLATE.format(key, _row)

    print(LATEX_TABLE_TEMPLATE.format(_alingment, len(ids)+1, _header, _rows))

def _get_final_results(results_dict, verbose=1):
    for _cond_name in results_dict.keys():
        for _key in results_dict[_cond_name].keys():
            verbose and print("Resuming list: ", results_dict[_cond_name][_key])
            results_dict[_cond_name][_key] = np.median(results_dict[_cond_name][_key])
            verbose and print("to: ", results_dict[_cond_name][_key], end='\n\n')

    return results_dict

def cond_prediction_comparison(model_ids, x_hidden_layers, z_hidden_layers, repetitions=10, iterations_mlp=1250, 
                               latex_format=False, random_state=1, verbose=1):
    '''It compares the prediction of the conditional parameters using the original spectra and the latent space. A basic
    neural network is used for prediction. If more than one id is passed, their datasets must be the same, as well as 
    the conditional parameters
    
    Parameters
    ----------
    model_ids : str
        Identifiers from which the trained autoencoders and the data corresponding to them are obtained
    x_hidden_layers : int list
        The ith element represents the number of neurons in the ith hidden layer
    z_hidden_layers : int list
        The ith element represents the number of neurons in the ith hidden layer
    repetitions : int, default 1
        Number of repetitions of the prediction, if it is more than 1 sample as a result the average
    iterations_mlp : int, default 1500
        Maximum number of iterations to run for each MLP
    latex_format : bool, default False
        If True, generates the latex code for the results table
    random_state : int, default 1
        MLP initialization seed and train_test_split
    verbose : {0 o 1}, default 1
        Verbose
    '''
    
    print('-'*WIDTH)
    print('Conditional parameters analysis'.center(WIDTH))
    print('-'*WIDTH)
    
    # Load data
    _data_dict_list = list()
    _last_ids = None
    _last_cond_name_labels = None
    for id_ in model_ids:
        _data_dict = my_utils.load_data(id_=id_, base_dir=ROOT_FOLDER)
        
        # Check if conditional parameters exist
        if (len(_data_dict['cond_params_names']) == 0):
            print('THERE AREN\'T CONDITIONAL PARAMETERS!')
            return
        
        # Check if both the data ids and the cond_name_labels are the same (this should be enough)
        if _last_ids is not None: # Check one of the two parameters is enough
            if ((_last_ids != _data_dict['ids']).all() or (_last_cond_name_labels != _data_dict['cond_params_names']).all()):
                print('Training/test data must be the same')
                return

        _last_ids = _data_dict['ids']
        _last_cond_name_labels = _data_dict['cond_params_names']

        _data_dict_list.append(_data_dict)


    _results = _initialize_results_dict(ids=model_ids, name_labels=_data_dict_list[0]['cond_params_names'])

    for repetition in range(repetitions):
        verbose and print('Repetition {:}:'.format(repetition+1))
        # The prediction of the parameters from the original data is only done once (per repetition) (because it is 
        # always the same data)
        _first_iteration = True 

        for _id, _data_dict in zip(model_ids, _data_dict_list):
            # Split data in train and test
            _X_train, _X_test, _y_train, _y_test, _z_train, _z_test = train_test_split(_data_dict['X'],
                                                                                    _data_dict['cond_params'],
                                                                                    _data_dict['z'],
                                                                                    random_state=random_state,
                                                                                    test_size=0.4)
            
            _cond_name_labels = _data_dict['cond_params_names']
            
            # For each conditional parameter
            for i, _label in enumerate(_cond_name_labels):
                    
                verbose and print('Prediction of parameter {}'.format(_label))
                verbose and print('Training models... ', end='', flush=True)
                
                ## If it is the first iteration, we also predicte from the original data
                if _first_iteration:
                    ### Creation and training of MLP to work with the original data
                    _regr_X = MLPRegressor(random_state=random_state, hidden_layer_sizes=x_hidden_layers, 
                                        max_iter=iterations_mlp, early_stopping=True, 
                                        verbose=False).fit(_X_train, _y_train[:, i])
                    ### Prediction
                    _regr_X_score = _regr_X.score(_X_test, _y_test[:, i])

                    if _regr_X_score > 0:

                        ## Save results
                        _results[_label]['regr_X_score'].append(_regr_X_score)
                    
                
                ## Creation and training of MLP to work with the latent space
                _regr_z = MLPRegressor(random_state=random_state, hidden_layer_sizes=z_hidden_layers, 
                                    max_iter=iterations_mlp, early_stopping=True, 
                                    verbose=False).fit(_z_train, _y_train[:, i]) 
                verbose and print('Done.')
                
                ## Prediction
                _regr_z_score = _regr_z.score(_z_test, _y_test[:, i])

                if (_regr_z_score) > 0:
                    ## Save results
                    _results[_label][_id+'_regr_z_score'].append(_regr_z_score)

            _first_iteration = False
            verbose and print()

    _get_final_results(_results, verbose=verbose)
        
    # Display results
    if not latex_format: 
        _print_plain_table(ids=model_ids, results_dict=_results, keys=_cond_name_labels) 
    else: 
        _print_latex_table(ids=model_ids, results_dict=_results, keys=_cond_name_labels)


def no_cond_prediction_comparison(model_ids, x_hidden_layers, z_hidden_layers, repetitions=1, iterations_mlp=1250, 
                                  latex_format=False, random_state=1, verbose=1):
    '''It compares the prediction of the non-conditional parameters using the original spectra and the latent space. A 
    basic neural network is used for prediction. If more than one id is passed, their datasets must be the same, as well
    as the conditional parameters.
    
    Parameters
    ----------
    model_ids : str
        Identifiers from which the trained autoencoders and the data corresponding to them are obtained
    x_hidden_layers : int list
        The ith element represents the number of neurons in the ith hidden layer
    z_hidden_layers : int list
        The ith element represents the number of neurons in the ith hidden layer
    repetitions : int, default 1
        Number of repetitions of the prediction, if it is more than 1 sample as a result the average
    iterations_mlp : int, default 1500
        Maximum number of iterations to run for each MLP
    latex_format : bool, default False
        If True, generates the latex code for the results table
    random_state : int, default 1
        MLP initialization seed and train_test_split
    verbose : {0 o 1}, default 1
        Verbose
    '''
    
    print('-'*WIDTH)
    print('Non-conditional parameters analysis'.center(WIDTH))
    print('-'*WIDTH)
    
    # Load data
    _data_dict_list = list()
    _last_ids = None
    _last_no_cond_name_labels = None
    for id_ in model_ids:
        _data_dict = my_utils.load_data(id_=id_, base_dir=ROOT_FOLDER)

        # Check if non-conditional parameters exist
        if (len(_data_dict['params_names']) == len(_data_dict['cond_params_names'])):
            print('THERE ARE NO NON-CONDITIONAL PARAMETERS!')
            return

        _no_cond_index = ~np.isin(_data_dict['params_names'], _data_dict['cond_params_names'])
        # Get non-conditional parameters
        _no_y_cond = _data_dict['params'][:, _no_cond_index]
        
        # Get non-conditional parameters names
        _no_cond_name_labels = _data_dict['params_names'][_no_cond_index]

        # Save non-conditional parameters names for later
        _data_dict['no_cond_name_labels'] = _no_cond_name_labels

        # Check if both the data ids and the cond_name_labels are the same (this should be enough)
        if _last_ids is not None: # Check one of the two parameters is enough
            if ((_last_ids != _data_dict['ids']).all() or (_last_no_cond_name_labels != _no_cond_name_labels).all()):
                print('Training/test data must be the same')
                return

        _last_ids = _data_dict['ids']
        _last_no_cond_name_labels = _no_cond_name_labels


        # Non-conditional paremeters normalization
        _qt = QuantileTransformer(random_state=random_state)
        _no_y_cond_normalized = _qt.fit_transform(_no_y_cond)

        # We add the normalized non-conditional parameters to the dictionary
        _data_dict['no_y_cond'] = _no_y_cond_normalized
        _data_dict_list.append(_data_dict)

    
    _results = _initialize_results_dict(ids=model_ids, name_labels=_data_dict_list[0]['no_cond_name_labels'])

    for repetition in range(repetitions):
        verbose and print('Repetition {:}:'.format(repetition+1))
        # The prediction of the parameters from the original data is only done once (per repetition) (because it is 
        # always the same data)
        _first_iteration = True 
    
        for _id, _data_dict in zip (model_ids, _data_dict_list):
            # Split data in train and test
            _X_train, _X_test, _y_train, _y_test, _z_train, _z_test = train_test_split(_data_dict['X'],
                                                                                    _data_dict['no_y_cond'],
                                                                                    _data_dict['z'],
                                                                                    random_state=random_state,
                                                                                    test_size=0.4)
            
            _no_cond_name_labels = _data_dict['no_cond_name_labels']

            # For each conditional parameter
            for i, _label in enumerate(_no_cond_name_labels):

                verbose and print('Prediction of parameter {}'.format(_label))
                verbose and print('Training models... ', end='', flush=True)

                ## If it is the first iteration, we also predicte from the original data
                if _first_iteration:
                    ### Creation and training of MLP to work with the original data
                    _regr_X = MLPRegressor(random_state=random_state, hidden_layer_sizes=x_hidden_layers, 
                                        max_iter=iterations_mlp, early_stopping=True, 
                                        verbose=False).fit(_X_train, _y_train[:, i])
                    
                    ### Prediction
                    _regr_X_score = _regr_X.score(_X_test, _y_test[:, i])

                        ## Save results
                    if _regr_X_score > 0:
                        _results[_label]['regr_X_score'].append(_regr_X_score)
                
                ## Creation and training of MLP to work with the latent space
                _regr_z = MLPRegressor(random_state=random_state, hidden_layer_sizes=z_hidden_layers, 
                                    max_iter=iterations_mlp, early_stopping=True, 
                                    verbose=False).fit(_z_train, _y_train[:, i]) 
                verbose and print('Done.')
                
                ## Prediction
                _regr_z_score = _regr_z.score(_z_test, _y_test[:, i])
                
                ## Save results
                if _regr_z_score > 0:
                    _results[_label][_id+'_regr_z_score'].append(_regr_z_score)

            _first_iteration = False
            verbose and print()

    _get_final_results(_results, verbose=verbose)
        
    # Display results
    if not latex_format:
        _print_plain_table(ids=model_ids, results_dict=_results, keys=_no_cond_name_labels)
    else: 
        _print_latex_table(ids=model_ids, results_dict=_results, keys=_no_cond_name_labels)


def models_comparison(model_ids, model_names=None, iterations=1250, repetitions=1, latex_format=False, 
                      export_graph=False, random_state=1, verbose=1):
    '''Compare the predictions of the conditional parameters of the different 
    models.
    
    Parameters
    ----------
    model_ids : str list
        List of ids of the models to analyze
    model_names : str list, default None
        List of model names to be analyzed
    iterations : int, default 1500
        Maximum number of iterations to run for each MLP
    repetitions : int, default 1
        Number of repetitions of the prediction, if it is more than 1 sample as a result the average
    latex_format : bool, default False
        If True, generates the latex code for the results table
    export_graph : bool, default False
        If True, export the graph to disk in png format
    random_state : int, default 1
        MLP initialization seed and train_test_split
    verbose : {0 o 1}, default 1
        Verbose
    '''
    def check_data(model_ids):
        _data_dict = my_utils.load_data(id_=model_ids[0], base_dir=ROOT_FOLDER)
        _X = _data_dict['X']
        _y = _data_dict['params']

        for _id in model_ids[1:]:
            _data_dict_to_compare = my_utils.load_data(id_=_id, base_dir=ROOT_FOLDER)
            _X_to_compare = _data_dict_to_compare['X']
            _y_to_compare = _data_dict_to_compare['params']
            if (_X != _X_to_compare).any() or (_y != _y_to_compare).any():
                return False

        return True

    # If no names are passed for the models, their ids are assigned
    if model_names is None:
        model_names = model_ids

    # It is verified that the datasets are equal
    if not check_data(model_ids=model_ids):
        print('X and params must be the same in all datasets')
        return
    
    _width = WIDTH
    for model_id, model_name in zip(model_ids, model_names):
        # Se obtiene información sobre el modelo del csv de resultados
        _results_df = pd.read_csv(os.path.join(ROOT_FOLDER, 'results/results.csv'))
        _rows_df = _results_df.loc[_results_df.model_id == model_id, ['discretize',
                                                                       'nbins',
                                                                       'cond_labels',
                                                                       'lambda_values']]
        _discretize, _nbins, _cond_aps, _lambda_ = _rows_df.iloc[0].values
        print()    
        print('-'*_width)
        print(('Model: {} (Discretize: {} / nbins: {} / cond_aps: {} / lambda: {})'.format(model_name, 
                                                                                           _discretize, 
                                                                                           _nbins if _discretize else '-',
                                                                                           _cond_aps, 
                                                                                           _lambda_).center(_width)))
        print('-'*_width)
        tsne_comparison(id_=model_id, cache='tests/tsne', random_state=random_state,
                        name=model_name, export_graph=export_graph, verbose=verbose)
    
    cond_prediction_comparison(model_ids=model_ids, iterations_mlp=iterations, repetitions=repetitions, 
                               latex_format=latex_format, random_state=random_state, verbose=verbose)
    no_cond_prediction_comparison(model_ids=model_ids, iterations_mlp=iterations, repetitions=repetitions, 
                                  latex_format=latex_format, random_state=random_state, verbose=verbose)
        
        
# ---
def cli():
    global ROOT_FOLDER

     # Parser
    # ## Parser definition
    parser = argparse.ArgumentParser(prog='test_cli', 
                                    description='Test disentanglement of the models through several test')

    # ## Parser arguments
    parser.add_argument('ids', metavar='id', type=str, nargs='+',
                        help='model ids to load the data generated during training')
    parser.add_argument('--model_names', metavar='name', type=str, nargs='+',
                        help='model names to be visualized with the results. The length must match the number of ids')
    parser.add_argument('--seed', type=int,
                        help='Value to replicate the results')
    parser.add_argument('--root_folder', type=str, default=os.path.dirname(__file__),
                                            help='root folder to save and load data')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbosity')

    test_group = parser.add_argument_group('Feasible tests')
    test_group.add_argument('--all', action='store_true',
                            help='run all tests. If this parameter is passed, the remain parameters are ignored')
    test_group.add_argument('--tsne_comparison', action='store_true',
                            help='comparison of spectra and latent space using the tsne algorithm')
    test_group.add_argument('--conditional_parameters_analysis', action='store_true',
                            help='analyze the result of predict the conditional parameters from the spectra vs from the latent space')
    test_group.add_argument('--no_conditional_parameters_analysis', action='store_true',
                            help='analyze the result of predict no conditional parameters from the spectra vs from the latent space')
    test_group.add_argument('--models_comparison', action='store_true',
                            help='compare a list of models')

    test_parameters_group = parser.add_argument_group('Test parameters')
    test_parameters_group.add_argument('--repetitions', type=int, default=1,
                                    help='Repetition of each training in cond and non-cond parameter analysis')
    test_parameters_group.add_argument('--mlp_max_iter', type=int, default=600,
                                        help='Maximun number of iterations of used MLPs. Default 600 iterations')
    test_parameters_group.add_argument('--x_hidden_layer_sizes', type=int, nargs='*', default=[200, 100], metavar='N',
                            help='The ith element represents the number of neurons in the ith hidden layer')
    test_parameters_group.add_argument('--z_hidden_layer_sizes', type=int, nargs='*', default=[200, 100], metavar='N',
                            help='The ith element represents the number of neurons in the ith hidden layer')
    test_parameters_group.add_argument('--latex_format', action='store_true',
                                    help='generate results in latex table format')

    tsne_parameters_group = parser.add_argument_group('Tsne parameters')                                    
    tsne_parameters_group.add_argument('--export_graph', action='store_true',
                                    help='Export a png graph with the comparisons')                                

    # ## Running the parser
    args = parser.parse_args()

    # ## Check conditions
    model_ids = args.ids
    model_names = args.model_names

    if model_names is not None:
        if len(model_names) != len(model_ids):
            raise ValueError("Model names and model ids have to have the same length")

    ROOT_FOLDER = args.root_folder

    # ## Get parameters
    VERBOSE = args.verbose    

    VERBOSE and print('\n-----------')
    VERBOSE and print('Parser info')
    VERBOSE and print('-----------')


    RANDOM_STATE = args.seed
    VERBOSE and print('Model ids: {:}'.format(model_ids))
    VERBOSE and print('Seed: {:}'.format(RANDOM_STATE), end='\n\n')

    ALL = args.all
    TSNE_COMPARISON = args.tsne_comparison
    COND_PARAM_ANALYSIS = args.conditional_parameters_analysis
    NO_COND_PARAM_ANALYSIS = args.no_conditional_parameters_analysis
    MODELS_COMPARISON = args.models_comparison
    VERBOSE and print('Test types')
    VERBOSE and print('---------------')
    VERBOSE and print('All: {:}'.format(ALL))
    VERBOSE and print('Tsne comparison: {:}'.format(TSNE_COMPARISON))
    VERBOSE and print('Conditional parameters analysis: {:}'.format(COND_PARAM_ANALYSIS))
    VERBOSE and print('No conditional parameters analysis: {:}'.format(NO_COND_PARAM_ANALYSIS))
    VERBOSE and print('Models comparison: {:}'.format(MODELS_COMPARISON))

    REPETITIONS = args.repetitions
    MAX_ITER = args.mlp_max_iter
    X_HIDDEN_LAYER_SIZES = args.x_hidden_layer_sizes
    Z_HIDDEN_LAYER_SIZES = args.z_hidden_layer_sizes
    LATEX_FORMAT = args.latex_format
    VERBOSE and print('\nTest parameters')
    VERBOSE and print('---------------')
    VERBOSE and print('Repetitions: {:}'.format(REPETITIONS))
    VERBOSE and print('Maximum of iterations: {:}'.format(MAX_ITER))
    VERBOSE and print('Latex format: {:}'.format(LATEX_FORMAT))

    EXPORT_GRAPH = args.export_graph
    VERBOSE and print('\nTsne parameters')
    VERBOSE and print('---------------')
    VERBOSE and print('Export graph: {:}'.format(EXPORT_GRAPH))

    # ---
    if ALL:
        TSNE_COMPARISON = COND_PARAM_ANALYSIS = NO_COND_PARAM_ANALYSIS = True
        
    if MODELS_COMPARISON:
        models_comparison(model_ids=model_ids,
                        model_names=model_names,
                        iterations=MAX_ITER,
                        repetitions=REPETITIONS,
                        random_state=RANDOM_STATE,
                        verbose=VERBOSE)
    else:
        if model_names is None:
            model_names = model_ids
        for _id, _name in zip(model_ids, model_names):
            print(('-'*(WIDTH//2)).center(WIDTH))
            print('Tests of model \'{:}\''.format(_name).center(WIDTH))
            print(('-'*(WIDTH//2)).center(WIDTH))
            
            if TSNE_COMPARISON:
                tsne_comparison(id_=_id, name=_name, cache='tests/tsne', random_state=RANDOM_STATE,
                                export_graph=EXPORT_GRAPH, verbose=VERBOSE)

        if NO_COND_PARAM_ANALYSIS:
            no_cond_prediction_comparison(model_ids=model_ids, 
                                          x_hidden_layers=X_HIDDEN_LAYER_SIZES, z_hidden_layers=Z_HIDDEN_LAYER_SIZES,
                                          repetitions=REPETITIONS, iterations_mlp=MAX_ITER, 
                                          latex_format=LATEX_FORMAT, random_state=RANDOM_STATE, verbose=VERBOSE)
                                            

        if COND_PARAM_ANALYSIS:
            cond_prediction_comparison(model_ids=model_ids, 
                                       x_hidden_layers=X_HIDDEN_LAYER_SIZES, z_hidden_layers=Z_HIDDEN_LAYER_SIZES,
                                       repetitions=REPETITIONS, iterations_mlp=MAX_ITER, 
                                       latex_format=LATEX_FORMAT, random_state=RANDOM_STATE, verbose=VERBOSE)

if __name__ == '__main__':
    cli()
# ---
