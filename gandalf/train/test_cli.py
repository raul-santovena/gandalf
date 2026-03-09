#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pandas as pd

import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.normpath('../../'))) # To support the use of the tool without packaging
import gandalf.train.utils as my_utils
from gandalf.test.tsne import tsne_comparison
from gandalf.test.cond_params import cond_prediction_comparison, no_cond_prediction_comparison

# # Internal constants
# ##
WIDTH = 100

def models_comparison(root_folder, model_ids, model_names=None, iterations=1250, repetitions=1, latex_format=False,
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
        _data_dict = my_utils.load_data(id_=model_ids[0], base_dir=root_folder)
        _X = _data_dict['X']
        _y = _data_dict['params']

        for _id in model_ids[1:]:
            _data_dict_to_compare = my_utils.load_data(id_=_id, base_dir=root_folder)
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
        # Information about the model is obtained from the results csv
        _results_df = pd.read_csv(os.path.join(root_folder, 'results/results.csv'))
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
        tsne_comparison(root_folder=root_folder, id_=model_id, cache='tests/tsne', random_state=random_state,
                        name=model_name, export_graph=export_graph, verbose=verbose)

    cond_prediction_comparison(root_folder=root_folder, model_ids=model_ids, iterations_mlp=iterations, repetitions=repetitions,
                               latex_format=latex_format, random_state=random_state, verbose=verbose)
    no_cond_prediction_comparison(root_folder=root_folder, model_ids=model_ids, iterations_mlp=iterations, repetitions=repetitions,
                                  latex_format=latex_format, random_state=random_state, verbose=verbose)


# ---
def cli():
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
    test_group.add_argument('--conditional_parameters_analysis', '--cond_params_analysis', action='store_true',
                            help='analyze the result of predict the conditional parameters from the spectra vs from the latent space')
    test_group.add_argument('--no_conditional_parameters_analysis', '--no_cond_params_analysis', action='store_true',
                            help='analyze the result of predict no conditional parameters from the spectra vs from the latent space')
    test_group.add_argument('--models_comparison', action='store_true',
                            help='compare a list of models')

    param_analysis_group = parser.add_argument_group('Parameters analysis options')
    param_analysis_group.add_argument('--repetitions', type=int, default=1,
                                      help='Repetition of each training in cond and non-cond parameter analysis')
    param_analysis_group.add_argument('--mlp_max_iter', type=int, default=600,
                                      help='Maximun number of iterations of used MLPs. Default 600 iterations')
    param_analysis_group.add_argument('--x_hidden_layer_sizes', type=int, nargs='*', default=[200, 100], metavar='N',
                                      help='The ith element represents the number of neurons in the ith hidden layer')
    param_analysis_group.add_argument('--z_hidden_layer_sizes', type=int, nargs='*', default=[200, 100], metavar='N',
                                      help='The ith element represents the number of neurons in the ith hidden layer')
    param_analysis_group.add_argument('--latex_format', action='store_true',
                                      help='generate results in latex table format')

    tsne_group = parser.add_argument_group('Tsne options')
    tsne_group.add_argument('--export_graph', action='store_true',
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
        models_comparison(root_folder=ROOT_FOLDER, model_ids=model_ids,
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
                tsne_comparison(root_folder=ROOT_FOLDER, id_=_id, name=_name, cache='tests/tsne', random_state=RANDOM_STATE,
                                export_graph=EXPORT_GRAPH, verbose=VERBOSE)

        if NO_COND_PARAM_ANALYSIS:
            no_cond_prediction_comparison(root_folder=ROOT_FOLDER, model_ids=model_ids,
                                          x_hidden_layers=X_HIDDEN_LAYER_SIZES, z_hidden_layers=Z_HIDDEN_LAYER_SIZES,
                                          repetitions=REPETITIONS, iterations_mlp=MAX_ITER,
                                          latex_format=LATEX_FORMAT, random_state=RANDOM_STATE, verbose=VERBOSE)


        if COND_PARAM_ANALYSIS:
            cond_prediction_comparison(root_folder=ROOT_FOLDER, model_ids=model_ids,
                                       x_hidden_layers=X_HIDDEN_LAYER_SIZES, z_hidden_layers=Z_HIDDEN_LAYER_SIZES,
                                       repetitions=REPETITIONS, iterations_mlp=MAX_ITER,
                                       latex_format=LATEX_FORMAT, random_state=RANDOM_STATE, verbose=VERBOSE)

if __name__ == '__main__':
    cli()
# ---
