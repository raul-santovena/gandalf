#!/usr/bin/env python
# coding: utf-8

# # Imports
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import datetime
from pickle import dump
from inspect import signature
import argparse

# My modules
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.normpath('../../'))) # To support the use of the tool without packaging
import gandalf.train.models as my_models
import gandalf.train.utils as my_utils
from gandalf.train.data_preparation import DataLoaderFactory

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


tf.keras.backend.set_floatx('float64')

# # Global variables
RESULTS_FILE_PATH = 'results/results.csv'
encoder = decoder = autoencoder = disc_models_dict = loss_fn_reconstruction = loss_fn_disc = optimizer_ae = optimizer_disc = ckpt = ckpt_manager = ROOT_FOLDER = MULTI_DISC = CONV_DISC = LAMBDA_DICT = COND_LABELS = DISCRETIZE = None


# # Utils
def get_model_command_line(model_id, results_file_path, verbose=0):
    '''Build the command line instruction of the trained model from the results file

    Parameters
    ----------
    model_id : str
        Model id
    '''
    _real_path = os.path.join(ROOT_FOLDER, results_file_path)
    verbose and print('Getting model {:} configuration from {:}'.format(model_id, _real_path))

    _df = pd.read_csv(_real_path)
    
    _rows = _df.loc[_df.model_id == model_id]
    if _rows.shape[0] == 0:
        raise ValueError("The model id {:} doesn't exist".format(model_id))
    
    _row = _rows.iloc[-1]
    args_dict = _row.to_dict()


    args_dict['survey_data_path'] = os.path.join(_row.dataset_dir, _row.dataset_name)
    args_dict['labels'] = _row.labels.replace('-',' ')
    args_dict['cond_labels'] = _row.cond_labels.replace('-', ' ')
    args_dict['lambda_values'] = _row.lambda_values.replace("[","").replace("]","").replace(",","")
    args_dict['encoder_hidden_layers'] = _row.encoder_hidden_layers.replace("[","").replace("]","").replace(",","")
    args_dict['decoder_hidden_layers'] = _row.decoder_hidden_layers.replace("[","").replace("]","").replace(",","")
    args_dict['disc_hidden_layers'] = _row.disc_hidden_layers.replace("[","").replace("]","").replace(",","")


    model_command_line = ('train_cli.py --training_id {model_id} ' +
    '--data_loader {data_loader} --survey_data_path {survey_data_path} --root_folder {root_folder} ' + 
    '--params {labels} --cond_params {cond_labels} ' +
    # Data parameters
    '{_normalize}{_shuffle}{_discretize}--nbins {nbins} --batch_size {batch_size} {_seed}' +
    # Model parameters
    '{_input_without_params} --latent_size {latent_size} ' +
    '--encoder_hidden_layer_sizes {encoder_hidden_layers} --decoder_hidden_layer_sizes {decoder_hidden_layers} ' +
    '--discriminator_hidden_layer_sizes {disc_hidden_layers} ' +
    '{_batch_norm}{_conv_disc}{_multi_disc}' +
    # Training parameters
    '--epochs {epochs} --lambda_values {lambda_values} {_dynamic_learning}' +
    '--discriminator_learning_rate {lr_disc} --autoencoder_learning_rate {lr_ae} ' +
    '-vv\n').format(_normalize="--normalize " if args_dict["X_normalization"] else "",
                   _shuffle="--shuffle " if args_dict["shuffle"] else "",
                   _discretize="--discretize " if args_dict["discretize"] else "",
                   _input_without_params="--input_without_params " if args_dict["input_without_params"] else "",
                   _batch_norm="--batch_normalization " if args_dict["batch_norm"] else "",
                   _conv_disc="--convolutional_discriminator " if args_dict["conv_disc"] else "",
                   _multi_disc="--multi_discriminator " if args_dict["multi_disc"] else "",
                   _dynamic_learning="--dynamic_learning {:} {:} ".format(args_dict['dl_no_change'], args_dict['dl_lambda_steps']) if args_dict["dynamic_learning"] else "",
                   _seed='--seed {:} '.format(args_dict['seed']) if (args_dict['seed'] != 'None') else '',
                   **args_dict)

    verbose and print('You can use this command to continue the training using the same settings (or vary some of them):')
    print(model_command_line)
# ---

# # Train functions
@tf.function
def train_step(x, y, *y_discs, lambda_dict):
    with tf.GradientTape() as tape_ae, tf.GradientTape(persistent=True) as tape_disc:
        # Obtaining latent space and autoencoder reconstruction
        z = encoder([x, y], training=True)
        output_ae = decoder([z, y], training=True)

        # Autoencoder reconstruction error calculation
        loss_reconstruction = loss_fn_reconstruction(x, output_ae)
        # Partial autoencoder error calculation
        loss_ae = loss_reconstruction 

        # Obtaining the prediction of each discriminator and its error
        loss_disc_dict = dict()
        for (disc_name, discriminator), y_disc in zip(disc_models_dict.items(), y_discs):
            # Obtaining discriminator predictions
            output_disc = discriminator(z if not CONV_DISC else z[:,:,np.newaxis], training=True)

            # Discrimnator error calculation
            loss_disc_dict[disc_name] = loss_fn_disc(y_disc, output_disc)

            # Calculation of the total error of the autoencoder applying the error of each discriminator to perform the adversarial training
            _weighted_lambda = lambda_dict[disc_name.replace('_discriminator','')] if MULTI_DISC else list(lambda_dict.values())[0]
            _loss_disc = -loss_disc_dict[disc_name]
            loss_ae += _weighted_lambda*_loss_disc
        
    # Calculation and application of gradients to modify the weights of the models
    for disc_name, discriminator in disc_models_dict.items():
        grads_disc = tape_disc.gradient(loss_disc_dict[disc_name], discriminator.trainable_weights)
        optimizer_disc.apply_gradients(zip(grads_disc, discriminator.trainable_weights))

    grads_ae = tape_ae.gradient(loss_ae, autoencoder.trainable_weights)
    optimizer_ae.apply_gradients(zip(grads_ae, autoencoder.trainable_weights))
    
    return loss_reconstruction, loss_ae, loss_disc_dict

def train(dataset, model_id=None, epochs=100, 
          search_ckpt=False, ckpt_step=10,
          saved_model_path='results/models',
          dynamic_learning=False, dl_no_change=None, dl_lambda_steps=None, 
          verbose=1):
    '''
    Perform model training.
    
    Parameters
    ----------
    dataset : tf.data.dataset
        Training dataset
    model_id : str, default None
        Training identifier used to generate log files for tensorboard
    search_ckpt : bool, default False
        Searches if there is a checkpoint for the current training, and loads it
    ckpt_step : int, default 10
        Creates a checkpoint every x epochs
    saved_model_path : str, default 'results/models'
        Saves the model at the end of training in the path specified by parameters, if it is None, the model is not saved
    dynamic_learning : bool, default False
        Perform a dynamic training that gradually increases the lambda value
    verbose : {0, 1 o 2}, default 1
        Verbose mode. 0 does not show output on screen, 1 output updated by epoch, 2 output updated by batch step
    '''
    
    # Generate id (if not passed by parameter) for the logs, checkpoint and final models
    if model_id is None:
        _model_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        _model_id = model_id
        
    # If there is a checkpoint, the last one is recover
    if search_ckpt and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Checkpoint restored from {}'.format(ckpt_manager.latest_checkpoint))
    else:
        print('No checkpoint found.')
        
            
    # Logs for tensorboard
    log_dir = os.path.join(ROOT_FOLDER, "logs/", "fit/", _model_id)
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    # Metrics
    epoch_loss_reconstruction_mean = tf.keras.metrics.Mean('epoch_loss_reconstruction_mean', dtype=tf.float32)
    epoch_loss_ae_mean = tf.keras.metrics.Mean('epoch_loss_ae_mean', dtype=tf.float32)
    epoch_loss_disc_mean_dict = dict()
    for _key in disc_models_dict.keys():
        epoch_loss_disc_mean_dict[_key] = tf.keras.metrics.Mean('epoch_loss_disc_{:}_mean'.format(_key), dtype=tf.float32)

    # Dynamic Learning
    _dl_marker = ''
    if dynamic_learning:
        _lambda_max = LAMBDA_DICT
        # A dictionary is created for the current lambda values and another for the increment values
        _lambda = dict()
        _lambda_rate = dict()
        for cond_label in COND_LABELS:
            _lambda[cond_label] = 0
            _lambda_rate[cond_label] = _lambda_max[cond_label]/dl_lambda_steps

        _epochs_no_change = dl_no_change
        _current_epochs_no_change = 0
        _best_ae_loss = sys.float_info.max
    else:
        _lambda = LAMBDA_DICT
    # ----------------
    
    # Epochs
    for _ in range(epochs):
        # Checkpoint epoch value is increased
        ckpt.epoch.assign_add(1)
        
        # The real epoch is obtained by checking the checkpoint
        real_epoch = int(ckpt.epoch)
        
        if verbose != 0:
            print("\nEpoch", real_epoch, end='\r')
            
        # Error metrics are reset
        epoch_loss_ae_mean.reset_states()
        for _key in disc_models_dict.keys():
            epoch_loss_disc_mean_dict[_key].reset_states()
        epoch_loss_reconstruction_mean.reset_states()

        # Unimproved iteration marker is created (dynamic_learning)
        if dynamic_learning:
            _dl_marker = '*'*_current_epochs_no_change
        
        time_start = time.time()
        # Batches
        for step, (x_batch_train, y_batch_train, *y_batch_train_encoded_list) in enumerate(dataset):
            
            # Training
            loss_reconstruction, loss_ae, loss_disc_dict = train_step(x_batch_train,
                                                                      y_batch_train,
                                                                      *y_batch_train_encoded_list if DISCRETIZE else y_batch_train,
                                                                      lambda_dict=_lambda)
            
            # Error metrics are updated
            epoch_loss_reconstruction_mean.update_state(loss_reconstruction)
            epoch_loss_ae_mean.update_state(loss_ae)
            verbose_loss_string = str()
            for _key in disc_models_dict.keys():
                epoch_loss_disc_mean_dict[_key].update_state(loss_disc_dict[_key])
                verbose_loss_string += '{:} loss: {:f} // '.format(_key, epoch_loss_disc_mean_dict[_key].result())

            if verbose == 2:
                print('Epoch {} - reconstruction loss: {:f} // Autoencoder loss: {:+f} // {:} (in {:.2f} seconds) {:}'.format(real_epoch,
                                                                                                                           epoch_loss_reconstruction_mean.result(), 
                                                                                                                           epoch_loss_ae_mean.result(),
                                                                                                                           verbose_loss_string, 
                                                                                                                           time.time() - time_start,
                                                                                                                           _dl_marker), end='\r')
                              
        
        if verbose == 1:
            print('Epoch {} - reconstruction loss: {:f} // Autoencoder loss: {:+f} // {:} (in {:.2f} seconds) {:}'.format(real_epoch,
                                                                                                                       epoch_loss_reconstruction_mean.result(), 
                                                                                                                       epoch_loss_ae_mean.result(),
                                                                                                                       verbose_loss_string,
                                                                                                                       time.time() - time_start,
                                                                                                                       _dl_marker), end='\r')
        # Dynamic learning
        if dynamic_learning:
            # If the autoencoder loss of this iteration is less than the best...
            if (epoch_loss_ae_mean.result() < _best_ae_loss):
                # The non-improvement iteration counter is reset
                _current_epochs_no_change = 0
                # Best loss (the lowest value)
                _best_ae_loss = epoch_loss_ae_mean.result()
            else:
                # If the autoencoder loss does not decrease, the non-improvement iteration counter is increased    
                _current_epochs_no_change += 1

            # If the iterations without improvement reach the marked limit, the lambda value is modified 
            if _current_epochs_no_change >= _epochs_no_change:
                verbose and print()
                _current_epochs_no_change = 0
                # The value of each lambda is increased
                for cond_label in COND_LABELS:
                    _old_lambda = _lambda[cond_label] # Verbose
                    _lambda[cond_label] += _lambda_rate[cond_label]
                    # If the maximum value defined for lambda is reached, the analysis is stopped
                    if _lambda[cond_label] >= _lambda_max[cond_label]:
                        _lambda[cond_label] = _lambda_max[cond_label]
                        dynamic_learning = False
                    verbose and print("{:} epochs without improvement. Lambda value for {:} discriminator has been modified from {:.6f} to {:.6f}".format(_epochs_no_change,
                                                                                                                                                            cond_label,
                                                                                                                                                            _old_lambda, _lambda[cond_label]))
                
        
        # Save checkpoint
        if real_epoch % ckpt_step == 0:
            ckpt_save_path = ckpt_manager.save(checkpoint_number=real_epoch)
            verbose and print('\nSaving checkpoint for epoch {} at {}'. format(real_epoch, os.path.abspath(ckpt_save_path)))
        
        # Log epoch metrics
        with summary_writer.as_default():
            tf.summary.scalar('reconstruction_loss', epoch_loss_reconstruction_mean.result(), real_epoch)
            tf.summary.scalar('autoencoder_loss', epoch_loss_ae_mean.result(), real_epoch)
            for _key in disc_models_dict.keys():
                tf.summary.scalar(_key+'_loss', epoch_loss_disc_mean_dict[_key].result(), real_epoch)
    
    # At the end, a last checkpoint is created (if it has not already done in the last epoch)
    if real_epoch % ckpt_step != 0:
        ckpt_save_path = ckpt_manager.save(checkpoint_number=real_epoch)
        verbose and print('\nSaving checkpoint for epoch {} at {}'.format(real_epoch, os.path.abspath(ckpt_save_path)))
    
    # At the end, the model is saved
    if saved_model_path is not None:
        _model_folder_path = os.path.normpath(os.path.join(ROOT_FOLDER, saved_model_path, model_id))
        ae_model_save_path = os.path.join(_model_folder_path, "autoencoder")
        autoencoder.save(ae_model_save_path)
        # All discriminator are saved
        for _name, _discriminator in disc_models_dict.items():
            _disc_model_save_path = os.path.join(_model_folder_path, _name)
            _discriminator.save(_disc_model_save_path)    
        verbose and print('\nSaving models at {}'.format(os.path.abspath(_model_folder_path)), end='\n\n')
# ---

# # Test functions
def eval_training(dataset_test, autoencoder, discriminators, verbose=1):
    '''Evaluation using the test dataset
    
    Parameters
    ----------
    dataset_test : Tensorflow.Dataset
        Test dataset
    autoencoder : Tensorflow.Model
        Autoencoder model
    discriminators : Tensorflow.Model dict
        Dict with the discriminators
    verbose : bool, default True
        Verbosity mode
    '''
    verbose and print('\nTest results')
    verbose and print('------------')
    
    # Data is extracted form the test dataset
    _X_test = np.concatenate([x for x, *_ in dataset_test], axis=0)
    _y_test = np.concatenate([y for _, y, *_ in dataset_test], axis=0)
    _y_test_encoded_list = np.concatenate([y_encoded for _, _, *y_encoded in dataset_test], axis=1)
    
    # Autoencoder reconstruction and autoencoder error (partial)
    _ae_reconstructions = autoencoder([_X_test, _y_test], training=False)
    _reconstruction_loss = loss_fn_reconstruction(_X_test, _ae_reconstructions)
    _ae_loss = _reconstruction_loss
    
    # We get the latent space
    _z = autoencoder.get_layer('encoder')([_X_test, _y_test], training=False)
    
    # Predictions and errors of the discriminators and final calculation of the autoencoder error
    _n_discs = len(discriminators.keys())
    _disc_loss_mean = 0
    for (_disc_name, _discriminator), _y_test_encoded in zip(discriminators.items(), _y_test_encoded_list if DISCRETIZE else _y_test):
        # Discriminator error
        _disc_predictions = _discriminator(_z, training=False)
        _disc_loss = loss_fn_disc(_y_test_encoded, _disc_predictions)
        verbose and print("{:}: {:f}".format(_disc_name, _disc_loss))
        _disc_loss_mean += _disc_loss
        
        # Calculation of the total error of the autoencoder applying the error of the discriminator
        _weighted_lambda = LAMBDA_DICT[_disc_name.replace('_discriminator','')]  if MULTI_DISC else list(LAMBDA_DICT.values())[0]
        _disc_loss = -_disc_loss
        _ae_loss += _weighted_lambda*_disc_loss
    _disc_loss_mean /= _n_discs
        
        
    
    # Final results
    verbose and print('Mean discriminator loss: {:f}'.format(_disc_loss_mean))
    verbose and print('\nReconstruction loss: {:f}'.format(_reconstruction_loss))
    verbose and print('Autoencoder loss: {:f}'.format(_ae_loss))
    
    return _ae_loss.numpy(), _reconstruction_loss.numpy(), _disc_loss_mean.numpy()
# ---

# # Results function
def generate_new_result_row(file_path, model_id, root_folder,
                            data_loader, dataset_dir, dataset_name,
                            dataset_rows, training_shape, test_shape,
                            shuffle, discretize, nbins, 
                            batch_norm, conv_disc, multi_disc,
                            X_normalization, params_normalization,
                            model_type, model_description, reconstruction_loss_name, discriminator_loss_name, 
                            opt_name_ae, lr_ae, opt_name_disc, lr_disc, seed,
                            encoder_hidden_layers, decoder_hidden_layers, 
                            input_without_params, latent_size,
                            disc_hidden_layers, ae_arquitecture, disc_arquitecture, 
                            labels, cond_labels, lambda_values, dynamic_learning, dl_no_change, dl_lambda_steps, 
                            epochs, batch_size,
                            reconstruction_loss, discriminator_loss, ae_loss, verbose):
    
    file_path = os.path.normpath(os.path.join(ROOT_FOLDER, file_path))

    num_params_in_csv = len(signature(generate_new_result_row).parameters) - 3 # -2 Because file_path and verbose won't be in the csv, an -1 because we need to add manually the last parameter without comma
    
    # if the file doesn't exit... create header
    if not os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write('model_id,root_folder,data_loader,dataset_dir,dataset_name,dataset_rows,training_shape,test_shape,shuffle,' +
                    'discretize,nbins,batch_norm,conv_disc,multi_disc,X_normalization,params_normalization,model_type,' +
                    'model_description,reconstruction_loss_name,discriminator_loss_name,opt_name_ae,lr_ae,opt_name_disc,' +
                    'lr_disc,seed,input_without_params,latent_size,encoder_hidden_layers,decoder_hidden_layers,' +
                    'disc_hidden_layers,ae_arquitecture,disc_arquitecture,labels,cond_labels,lambda_values,' +
                    'dynamic_learning,dl_no_change,dl_lambda_steps,epochs,batch_size,reconstruction_loss,' +
                    'discriminator_loss,ae_loss\n')
            
    # Add results row
    with open(file_path, 'a') as f:
        f.write((num_params_in_csv*'{:},'+'{:}\n').format(model_id, os.path.abspath(root_folder), data_loader, 
                                                          os.path.abspath(dataset_dir), dataset_name, dataset_rows, 
                                                          training_shape, test_shape, shuffle, discretize, nbins, 
                                                          batch_norm, conv_disc, multi_disc, X_normalization, 
                                                          params_normalization, model_type, model_description, 
                                                          reconstruction_loss_name, discriminator_loss_name, 
                                                          opt_name_ae, lr_ae, opt_name_disc, lr_disc, seed, 
                                                          input_without_params, latent_size, '"'+str(encoder_hidden_layers)+'"',
                                                          '"'+str(decoder_hidden_layers)+'"', 
                                                          '"'+str(disc_hidden_layers)+'"', 
                                                          ae_arquitecture, disc_arquitecture, labels, cond_labels, 
                                                          '"'+str(lambda_values)+'"', dynamic_learning, dl_no_change, 
                                                          dl_lambda_steps, epochs, batch_size, reconstruction_loss, 
                                                          discriminator_loss, ae_loss))
        
        verbose and print('\nSaving stats in {:}'.format(os.path.abspath(file_path)))
# ---

def cli():
    global encoder, decoder, autoencoder, disc_models_dict, loss_fn_reconstruction, loss_fn_disc, optimizer_ae 
    global optimizer_disc, ckpt, ckpt_manager, ROOT_FOLDER, MULTI_DISC, CONV_DISC, LAMBDA_DICT, COND_LABELS, DISCRETIZE

    # # Parser
    # ## Parser definition
    parser = argparse.ArgumentParser(prog='train_cli', description='Define and train disentangling models')

    # ## Parser arguments
    parser.add_argument('--data_loader', type=str,  default='SampleDataLoader', 
                        help='name of the data loader class')
    parser.add_argument('--survey_data_path', type=str,
                        help='Path where the data survey is located')
    parser.add_argument('--params', type=str, nargs='+', default=['teff', 'logg'],
                        help='parameters contained in the data')
    parser.add_argument('--cond_params', type=str, nargs='+', default=['teff'],
                        help='parameters that will be decomposed')
    parser.add_argument('--training_id', '--model_id', type=str,
                        help='model/training id to load a specific training, including data and models. If this parameter is passed, all parameters except the training ones are ignored (currently, is the user who has to pass the same parameters)')
    parser.add_argument('--get_model_settings', type=str, metavar='MODEL_ID',
                        help='return the command line instruction to continue the training of the specified id')    
    parser.add_argument('--root_folder', type=str, default=os.path.dirname(__file__),
                                            help='root folder to save and load data')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                             help='verbose')                    

    data_group = parser.add_argument_group('Data parameters')
    data_group.add_argument('--normalize', action='store_true',
                            help='Normalize X')
    data_group.add_argument('--shuffle', action='store_true',
                            help='shuffle data before the dataset is created')
    data_group.add_argument('--discretize', action='store_true',
                            help='discretizes the conditional parameters into bins for training')
    data_group.add_argument('--nbins', type=int, default=10, 
                            help='number of bins')
    data_group.add_argument('--batch_size', type=int, default=50, 
                            help='batch size')
    data_group.add_argument('--seed', type=int, 
                            help='seed to replicate the results')

    models_group = parser.add_argument_group('Models parameters')
    models_group.add_argument('--input_without_params', action='store_true',
                            help='the conditional parameters are only passed to latent space')
    models_group.add_argument('--latent_size', type=int, default=25,
                            help='size of the latent space in the autoencoder')
    models_group.add_argument('--encoder_hidden_layer_sizes', type=int, nargs='*', default=[512, 256], metavar='N',
                            help='The ith element represents the number of neurons in the ith hidden layer')
    models_group.add_argument('--decoder_hidden_layer_sizes', type=int, nargs='*', default=[256, 512], metavar='N',
                            help='The ith element represents the number of neurons in the ith hidden layer')
    models_group.add_argument('--discriminator_hidden_layer_sizes', type=int, nargs='*', default=[64, 32], metavar='N',
                            help='The ith element represents the number of neurons in the ith hidden layer')
    models_group.add_argument('--batch_normalization', '--batch_norm', action='store_true',
                            help='Add batch normalization between layers to both autoencoder and discriminators')
    models_group.add_argument('--convolutional_discriminator', '--conv_disc', action='store_true',
                            help='make a convolutional discriminator instead of regular ann layers')
    models_group.add_argument('--multi_disc', '--multi_discriminator', action='store_true',
                            help='create an architecture with a discriminator per conditional parameter')
    models_group.add_argument('--summary_models', '--summary', action='store_true',
                            help='print the summary of the models')

    training_group = parser.add_argument_group('Training parameters')
    training_group.add_argument('--epochs', type=int, default=200,
                                help='Number of epoch during training')
    training_group.add_argument('--lambda_values', type=float, nargs='+', default=[0.001],
                                help='Lambda values to control the disentanglement of each parameter')
    training_group.add_argument('--dynamic_learning', '--dynamic_lambda', 
                                type=int, nargs=2, metavar=('NO_CHANGE', 'STEPS'), default=[None, None],
                                help='use a dynamic lambda value during training')
    training_group.add_argument('--discriminator_learning_rate', '--disc_lr', type=float, default=0.001,
                                help='Learning rate of the discriminator')
    training_group.add_argument('--autoencoder_learning_rate', '--ae_lr', type=float, default=0.001,
                                help='Learning rate of the autoencoder')

    # ## Running the parser
    args = parser.parse_args()

    # ## Check essential parameters
    if args.get_model_settings:
        ROOT_FOLDER = args.root_folder
        get_model_command_line(model_id=args.get_model_settings,
                               results_file_path=RESULTS_FILE_PATH,
                               verbose=args.verbose)
        sys.exit(0)

    if (not args.multi_disc) and (len(args.lambda_values) != 1):
        raise ValueError("The number of lambdas must be match with the number of discriminators. To use several lambdas use the '--multi_disc' parameter")

    # ## Get parameters
    VERBOSE = args.verbose
    ROOT_FOLDER = args.root_folder

    DATA_LOADER = args.data_loader
    MODEL_ID = args.training_id
    SURVEY_DATA_PATH = args.survey_data_path
    LABELS = args.params
    COND_LABELS = args.cond_params
    VERBOSE and print('Training id: {:}'.format(MODEL_ID), end='\n\n')
    VERBOSE and print('Data loader: {:}'.format(DATA_LOADER), end='\n\n')
    VERBOSE and print('Survey data path: {:}'.format(SURVEY_DATA_PATH))
    VERBOSE and print('Parameters: {:}'.format(LABELS))
    VERBOSE and print('Conditional parameters: {:}'.format(COND_LABELS), end='\n\n')


    NORMALIZE_SPECTRA = args.normalize
    SHUFFLE = args.shuffle
    DISCRETIZE = args.discretize
    NBINS = args.nbins
    SEED = args.seed

    VERBOSE and print('Data parameters')
    VERBOSE and print('---------------')
    VERBOSE and print('Normalize spectra: {:}'.format(NORMALIZE_SPECTRA))
    VERBOSE and print('Shuffle data: {:}'.format(SHUFFLE))
    VERBOSE and print('Discretize conditional parameters: {:}'.format(DISCRETIZE))
    VERBOSE and print('Number of bins if discretization is true: {:}'.format(NBINS))
    VERBOSE and print('Seed value: {:}'.format(SEED))

    PARAMS_DIM = len(COND_LABELS) 
    INPUT_WITHOUT_PARAMS = args.input_without_params
    LATENT_SIZE = args.latent_size
    ENCODER_HIDDEN_LAYER_SIZES = args.encoder_hidden_layer_sizes
    DECODER_HIDDEN_LAYER_SIZES = args.decoder_hidden_layer_sizes
    DISC_HIDDEN_LAYER_SIZES = args.discriminator_hidden_layer_sizes
    BATCH_NORM = args.batch_normalization
    CONV_DISC = args.convolutional_discriminator
    MULTI_DISC = args.multi_disc
    SUMMARY_MODELS = args.summary_models
    VERBOSE and print('Model parameters')
    VERBOSE and print('----------------')
    VERBOSE and print('Input without params: {:}'.format(INPUT_WITHOUT_PARAMS))
    VERBOSE and print('Latent size: {:}'.format(LATENT_SIZE))
    VERBOSE and print('Size of hidden layers of the encoder: {:}'.format(ENCODER_HIDDEN_LAYER_SIZES))
    VERBOSE and print('Size of hidden layers of the decoder: {:}'.format(DECODER_HIDDEN_LAYER_SIZES))
    VERBOSE and print('Size of hidden layers of the discriminator(s): {:}'.format(DISC_HIDDEN_LAYER_SIZES))
    VERBOSE and print('Batch normalization: {:}'.format(BATCH_NORM))
    VERBOSE and print('Convolutional discriminator: {:}'.format(CONV_DISC))
    VERBOSE and print('Multi-discriminators: {:}'.format(MULTI_DISC))
    VERBOSE and print('Summary of the models: {:}'.format(SUMMARY_MODELS), end='\n\n')


    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LAMBDA_LIST = args.lambda_values
    if (len(LAMBDA_LIST) == 1) and MULTI_DISC:
        LAMBDA_LIST[0] = LAMBDA_LIST[0]/PARAMS_DIM
        LAMBDA_LIST *= PARAMS_DIM
    LAMBDA_DICT = dict(zip(COND_LABELS, LAMBDA_LIST))
    DL_NO_CHANGE, DL_LAMBDA_STEPS = args.dynamic_learning
    DYNAMIC_LEARNING = True if (DL_NO_CHANGE is not None) and (DL_LAMBDA_STEPS is not None) else False
    LR_DISC = args.discriminator_learning_rate
    LR_AE = args.autoencoder_learning_rate
    VERBOSE and print('Training parameters')
    VERBOSE and print('-------------------')
    VERBOSE and print('Epochs: {:}'.format(EPOCHS))
    VERBOSE and print('Batch size: {:}'.format(BATCH_SIZE))
    VERBOSE and print('Lambda: {:}'.format(LAMBDA_LIST))
    VERBOSE and print('Epochs no change (Dynamic learning): {:}'.format(DL_NO_CHANGE))
    VERBOSE and print('Lambda steps (Dynamic learning): {:}'.format(DL_LAMBDA_STEPS))
    VERBOSE and print('Discriminator learning rate: {:}'.format(LR_DISC))
    VERBOSE and print('Autoencoder learning rate: {:}'.format(LR_AE))

    # # Data
    VERBOSE and print('\nBuilding datasets...')
    dataLoader = DataLoaderFactory().create_dataloader(data_loader=DATA_LOADER, dataset_path=SURVEY_DATA_PATH, params=LABELS, 
                                                    conditional_params=COND_LABELS, normalize=NORMALIZE_SPECTRA, shuffle=SHUFFLE, 
                                                    discretize=DISCRETIZE, nbins=NBINS, random_state=SEED, verbose=VERBOSE)
                                
    # Get train and test tensorflow datasets
    dataset_train, dataset_test = dataLoader.create_tf_dataset(batch=BATCH_SIZE,
                                                            multi_discriminator=MULTI_DISC)
    VERBOSE and print('Done')

    # Get input/outpout dimensions from datasets
    DATA_DIM = dataset_train.element_spec[0].shape[1]
    OUTPUT_DISCRIMINATOR = dataset_train.element_spec[2].shape[1] if DISCRETIZE else PARAMS_DIM


    # # Models

    # ## Losses

    # Training losses definition
    loss_fn_disc = tf.keras.losses.CategoricalCrossentropy() if DISCRETIZE else tf.keras.losses.MeanSquaredError()
    loss_fn_reconstruction = tf.keras.losses.MeanSquaredError()


    # Losses names are obtained for the results file
    loss_disc_name = loss_fn_disc.get_config()['name']
    loss_reconstruction_name = loss_fn_reconstruction.get_config()['name']

    # ---

    # ## Model definitions

    encoder = None
    decoder = None
    autoencoder = None
    disc_models_dict = dict() # If the --multi_discriminator parameter is not provided, the list will have just one discriminator

    # Check whether a trained model has to be loaded. If it hasn't, the models are created
    if MODEL_ID is None:
        VERBOSE and print('\nBuilding models', end='')
        SUMMARY_MODELS and print(' and showing their architectures...')

        model_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # If encoder/decoder are not initialized, the 'make_autoencoder_model' function creates them with the default parameters
        encoder = my_models.make_encoder_model(data_dim=DATA_DIM, 
                                            params_dim=PARAMS_DIM, 
                                            latent_dim=LATENT_SIZE,
                                            hidden_layer_sizes=ENCODER_HIDDEN_LAYER_SIZES, 
                                            hidden_layers_activation='relu',
                                            input_only_spectra=INPUT_WITHOUT_PARAMS,
                                            batch_norm=BATCH_NORM,
                                            verbose=SUMMARY_MODELS)

        decoder = my_models.make_decoder_model(data_dim=DATA_DIM, 
                                            params_dim=PARAMS_DIM, 
                                            latent_dim=LATENT_SIZE,
                                            hidden_layer_sizes=DECODER_HIDDEN_LAYER_SIZES,
                                            hidden_layers_activation='relu',
                                            batch_norm=BATCH_NORM,
                                            verbose=SUMMARY_MODELS)



        autoencoder = my_models.make_autoencoder_model(data_dim=DATA_DIM, params_dim=PARAMS_DIM,
                                                    latent_dim=LATENT_SIZE, 
                                                    encoder=encoder, 
                                                    decoder=decoder)

        
        if MULTI_DISC:
            for y_cond in COND_LABELS:
                _discriminator = my_models.make_discriminator(latent_dim=LATENT_SIZE, nbins=OUTPUT_DISCRIMINATOR,
                                                        hidden_layer_sizes=DISC_HIDDEN_LAYER_SIZES,
                                                        convolutional=CONV_DISC,
                                                        batch_norm=BATCH_NORM,
                                                        output_activation='sigmoid' if DISCRETIZE else 'linear',
                                                        verbose=SUMMARY_MODELS)
                disc_models_dict[y_cond+'_discriminator'] = _discriminator
        else:
            _discriminator = my_models.make_discriminator(latent_dim=LATENT_SIZE, nbins=OUTPUT_DISCRIMINATOR,
                                                    hidden_layer_sizes=DISC_HIDDEN_LAYER_SIZES,
                                                    convolutional=CONV_DISC,
                                                    batch_norm=BATCH_NORM,
                                                    output_activation='sigmoid' if DISCRETIZE else 'linear',
                                                    verbose=SUMMARY_MODELS)
            disc_models_dict['_'.join(COND_LABELS)+'_discriminator'] = _discriminator

        (not SUMMARY_MODELS) and VERBOSE and print('. Done')
        SUMMARY_MODELS and print('Models built')
    else:
        model_id = MODEL_ID
        autoencoder, disc_models_dict = my_utils.load_models(model_id=model_id, model_path=os.path.join(ROOT_FOLDER, 'results/models'), verbose=VERBOSE)
        encoder = autoencoder.get_layer('encoder')
        decoder = autoencoder.get_layer('decoder')

    # ---

    # # Training
    VERBOSE and print('\nThe training/model id is', model_id)

    # ## Optimizers
    optimizer_disc = tf.keras.optimizers.Adam(learning_rate=LR_DISC)
    optimizer_ae = tf.keras.optimizers.Adam(learning_rate=LR_AE)

    # ## Checkpoints
    checkpoint_path = os.path.join(ROOT_FOLDER, 'checkpoints', 'train', model_id)

    ckpt = tf.train.Checkpoint(autoencoder=autoencoder,
                            **disc_models_dict,
                            optimizer_ae=optimizer_ae,
                            optimizer_disc=optimizer_disc,
                            epoch=tf.Variable(0))


    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=checkpoint_path, max_to_keep=3)


    # ## Training
    train(dataset=dataset_train, model_id=model_id, epochs=EPOCHS,
        search_ckpt=True, ckpt_step=20,
        saved_model_path='results/models',
        dynamic_learning=DYNAMIC_LEARNING, dl_no_change=DL_NO_CHANGE, dl_lambda_steps=DL_LAMBDA_STEPS,
        verbose=VERBOSE)


    # ---


    # # Test
    ae_loss, reconstruction_loss, disc_loss = eval_training(dataset_test=dataset_test, 
                                                            autoencoder=autoencoder, 
                                                            discriminators=disc_models_dict,
                                                            verbose=True)
    # ---


    # # Results
    model_id = model_id

    # The data is extracted from the datasets 
    X_train = np.concatenate([x for x, *_ in dataset_train], axis=0)
    X_test = np.concatenate([x for x, *_ in dataset_test], axis=0)
    cond_params_train = np.concatenate([y for _, y, *_ in dataset_train], axis=0)
    cond_params_test = np.concatenate([y for _, y, *_ in dataset_test], axis=0)
    z_train = autoencoder.get_layer('encoder')([X_train, cond_params_train], training=False)
    z_test = autoencoder.get_layer('encoder')([X_test, cond_params_test], training=False)
    decoded_train = autoencoder([X_train, cond_params_train], training=False)
    decoded_test = autoencoder([X_test, cond_params_test], training=False)


    # model_id already defined
    # survey already defined as SURVEY
    dataset_dir = dataLoader.get_dataset_dir()
    dataset_name = dataLoader.get_dataset_name()
    dataset_rows = X_train.shape[0] + X_test.shape[0]
    training_shape = '{:}x{:}'.format(X_train.shape[0], X_train.shape[1])
    test_shape = '{:}x{:}'.format(X_test.shape[0], X_test.shape[1])

    # shuffle already defined as SHUFFLE
    # discretize already defined as DISCRETIZED
    # nbins already defined as NBINS
    # seed already defined as SEED
    # batch_norm defined as BATCH_NORM
    # conv_disc defined as CONV_DISC
    # multi_discriminator defined as MULTI_DISC
    X_normalization = 'Quantile Transformer' if NORMALIZE_SPECTRA else None
    params_normalization = 'Quantile Transformer'
    model_type = 'Adversarial CVAE'
    model_description = ''
    opt_name_ae = optimizer_ae.get_config()['name']
    # lr_ae already defined
    opt_name_disc = optimizer_disc.get_config()['name']
    # lr_disc already defined
    encoder_hidden_layers = ENCODER_HIDDEN_LAYER_SIZES
    decoder_hidden_layers = DECODER_HIDDEN_LAYER_SIZES
    # latent_size already defined as LATENT_SIZE
    ae_arquitecture = '{:}+{:}-{:}-{:}-{:}+{:}-{:}-{:}-{:}'.format(DATA_DIM, PARAMS_DIM,
                                                                ENCODER_HIDDEN_LAYER_SIZES[0], ENCODER_HIDDEN_LAYER_SIZES[1],
                                                                LATENT_SIZE, PARAMS_DIM,
                                                                DECODER_HIDDEN_LAYER_SIZES[0], DECODER_HIDDEN_LAYER_SIZES[1],
                                                                DATA_DIM)
    disc_hidden_layers = DISC_HIDDEN_LAYER_SIZES
    disc_arquitecture = '{:}-{:}-{:}-{:}'.format(LATENT_SIZE, 
                                                DISC_HIDDEN_LAYER_SIZES[0], DISC_HIDDEN_LAYER_SIZES[1], 
                                                OUTPUT_DISCRIMINATOR)

    # labels already defined as LABELS
    # cond_labels already defined as COND_LABELS
    # lambda_ = LAMBDA already defined
    # dynamic_learning = DYNAMIC_LEARNING already defined
    # dl_no_change = DL_NO_CHANGE already defined
    # dl_lambda_steps = DL_LAMBDA_STEPS already defined
    # epochs = EPOCHS already defined
    # batch_size = BATCH_SIZE already defined
    # reconstruction_loss already defined
    # discriminator_loss already defined
    # ae_loss already defined


    generate_new_result_row(file_path=RESULTS_FILE_PATH, model_id=model_id, root_folder=ROOT_FOLDER, data_loader=DATA_LOADER, dataset_dir=dataset_dir, 
                            dataset_name=dataset_name, dataset_rows=dataset_rows, training_shape=training_shape, 
                            test_shape=test_shape, shuffle=SHUFFLE, discretize=DISCRETIZE, nbins=NBINS, 
                            batch_norm=BATCH_NORM, conv_disc=CONV_DISC, multi_disc=MULTI_DISC, 
                            X_normalization=X_normalization, params_normalization=params_normalization, 
                            model_type=model_type, model_description=model_description, 
                            reconstruction_loss_name=loss_reconstruction_name, discriminator_loss_name=loss_disc_name,
                            opt_name_ae=opt_name_ae, lr_ae=LR_AE, opt_name_disc=opt_name_disc, lr_disc=LR_DISC, seed=SEED, 
                            encoder_hidden_layers=encoder_hidden_layers, decoder_hidden_layers=decoder_hidden_layers, 
                            input_without_params=INPUT_WITHOUT_PARAMS, latent_size=LATENT_SIZE, ae_arquitecture=ae_arquitecture, 
                            disc_hidden_layers=disc_hidden_layers, disc_arquitecture=disc_arquitecture, 
                            labels=str(LABELS).replace('[','').replace(']','').replace("'",'').replace(', ','-'),
                            cond_labels=str(COND_LABELS).replace('[','').replace(']','').replace("'",'').replace(', ','-'), 
                            lambda_values=LAMBDA_LIST, dynamic_learning=DYNAMIC_LEARNING, dl_no_change=DL_NO_CHANGE, 
                            dl_lambda_steps=DL_LAMBDA_STEPS, epochs=ckpt.epoch.numpy(), batch_size=BATCH_SIZE, 
                            reconstruction_loss=reconstruction_loss, discriminator_loss=disc_loss, ae_loss=ae_loss,
                            verbose=VERBOSE)


    # ## Save results for visual-autoencoder
    base_folder_path = 'results'
    data_folder = 'data'

    data_folder_path = os.path.normpath(os.path.join(ROOT_FOLDER, base_folder_path, data_folder, model_id))

    # Create folder if doesn't exist
    if not os.path.exists(data_folder_path):
        print('Saving data. Folder doesn\'t exist! Creating in \'{:}\''.format(os.path.abspath(data_folder_path)))
        os.makedirs(data_folder_path)
    else:
        print('Saving data in {:}'.format(os.path.abspath(data_folder_path)))


    y_labels = dataLoader.get_name_labels()

    # Saving data
    np.save(os.path.join(data_folder_path, 'X_train'), X_train)
    np.save(os.path.join(data_folder_path, 'X_test'), X_test)
    np.save(os.path.join(data_folder_path, 'cond_params_train'), cond_params_train)
    np.save(os.path.join(data_folder_path, 'cond_params_test'), cond_params_test)
    np.save(os.path.join(data_folder_path, 'params_train'), dataLoader.get_original_params_train())
    np.save(os.path.join(data_folder_path, 'params_test'), dataLoader.get_original_params_test())
    np.save(os.path.join(data_folder_path, 'z_train'), z_train)
    np.save(os.path.join(data_folder_path, 'z_test'), z_test)
    np.save(os.path.join(data_folder_path, 'decoded_train'), decoded_train)
    np.save(os.path.join(data_folder_path, 'decoded_test'), decoded_test)
    np.save(os.path.join(data_folder_path, 'ids_train'), dataLoader.get_ids_train())
    np.save(os.path.join(data_folder_path, 'ids_test'), dataLoader.get_ids_test())
    np.save(os.path.join(data_folder_path, 'axis_labels'), dataLoader.get_axis_labels())
    np.save(os.path.join(data_folder_path, 'params_names'), y_labels)
    np.save(os.path.join(data_folder_path, 'cond_params_names'), COND_LABELS)

    # Saving scalers
    dump(dataLoader.get_scaler_X() if NORMALIZE_SPECTRA else None, open(os.path.join(data_folder_path, 'X_scaler.pkl'), 'wb'))
    dump(dataLoader.get_scaler_params(), open(os.path.join(data_folder_path, 'param_scaler.pkl'), 'wb'))

if __name__ == '__main__':
    cli()
# ---
# ---