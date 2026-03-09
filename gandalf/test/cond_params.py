import sys
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer # For no conditional parameters

# My modules
sys.path.append("../") # To support the use of the tool without packaging
import gandalf.train.utils as my_utils

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

def _get_final_results(results_dict, verbose=1):
    for _cond_name in results_dict.keys():
        for _key in results_dict[_cond_name].keys():
            results_dict[_cond_name][_key] = np.median(results_dict[_cond_name][_key])

    return results_dict

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

def cond_prediction_comparison(model_ids, root_folder, x_hidden_layers, z_hidden_layers, repetitions=10, iterations_mlp=1250,
                               latex_format=False, random_state=1, verbose=1):
    '''It compares the prediction of the conditional parameters using the original spectra and the latent space. A basic
    neural network is used for prediction. If more than one id is passed, their datasets must be the same, as well as
    the conditional parameters

    Parameters
    ----------
    model_ids : str
        Identifiers from which the trained autoencoders and the data corresponding to them are obtained
    root_folder : str
        Path to the root folder where the data is located
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
    verbose : {0, 1, 2}, default 0
        Verbose, if 1, print messages, if 2 shows MLP progress
    '''

    print('-'*WIDTH)
    print('Conditional parameters analysis'.center(WIDTH))
    print('-'*WIDTH)

    show_progress = verbose == 2

    # Load data
    _data_dict_list = list()
    _last_ids = None
    _last_cond_name_labels = None
    for id_ in model_ids:
        _data_dict = my_utils.load_data(id_=id_, base_dir=root_folder)

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
                                        verbose=show_progress).fit(_X_train, _y_train[:, i])
                    ### Prediction
                    _regr_X_score = _regr_X.score(_X_test, _y_test[:, i])

                    ## Save results
                    _results[_label]['regr_X_score'].append(_regr_X_score)


                ## Creation and training of MLP to work with the latent space
                _regr_z = MLPRegressor(random_state=random_state, hidden_layer_sizes=z_hidden_layers,
                                    max_iter=iterations_mlp, early_stopping=True,
                                    verbose=show_progress).fit(_z_train, _y_train[:, i])
                verbose and print('Done.')

                ## Prediction
                _regr_z_score = _regr_z.score(_z_test, _y_test[:, i])

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

def no_cond_prediction_comparison(model_ids, root_folder, x_hidden_layers, z_hidden_layers, repetitions=1, iterations_mlp=1250,
                                  latex_format=False, random_state=1, verbose=1):
    '''It compares the prediction of the non-conditional parameters using the original spectra and the latent space. A
    basic neural network is used for prediction. If more than one id is passed, their datasets must be the same, as well
    as the conditional parameters.

    Parameters
    ----------
    model_ids : str
        Identifiers from which the trained autoencoders and the data corresponding to them are obtained
    root_folder : str
        Path to the root folder where the data is located
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
    verbose : {0, 1, 2}, default 0
        Verbose, if 1, print messages, if 2 shows MLP progress
    '''

    print('-'*WIDTH)
    print('Non-conditional parameters analysis'.center(WIDTH))
    print('-'*WIDTH)

    show_progress = verbose == 2

    # Load data
    _data_dict_list = list()
    _last_ids = None
    _last_no_cond_name_labels = None
    for id_ in model_ids:
        _data_dict = my_utils.load_data(id_=id_, base_dir=root_folder)

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
                                        verbose=show_progress).fit(_X_train, _y_train[:, i])

                    ### Prediction
                    _regr_X_score = _regr_X.score(_X_test, _y_test[:, i])

                    ## Save results
                    _results[_label]['regr_X_score'].append(_regr_X_score)

                ## Creation and training of MLP to work with the latent space
                _regr_z = MLPRegressor(random_state=random_state, hidden_layer_sizes=z_hidden_layers,
                                    max_iter=iterations_mlp, early_stopping=True,
                                    verbose=show_progress).fit(_z_train, _y_train[:, i])
                verbose and print('Done.')

                ## Prediction
                _regr_z_score = _regr_z.score(_z_test, _y_test[:, i])

                ## Save results
                _results[_label][_id+'_regr_z_score'].append(_regr_z_score)

            _first_iteration = False
            verbose and print()

    _get_final_results(_results, verbose=verbose)

    # Display results
    if not latex_format:
        _print_plain_table(ids=model_ids, results_dict=_results, keys=_no_cond_name_labels)
    else:
        _print_latex_table(ids=model_ids, results_dict=_results, keys=_no_cond_name_labels)
