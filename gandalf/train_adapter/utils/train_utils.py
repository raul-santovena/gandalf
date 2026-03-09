import os
import time
import numpy as np
import tensorflow as tf
import sys
import datetime

tf.keras.backend.set_floatx('float64')

'''
These functions are obtained from train_cli in the original GANDALF code.
In this script there is a problem with global variables, it uses parameters and this type of variables, making the code difficult to adapt to this library version.
The best solution would be to refactor the code, but this is time consuming and can introduce unexpected bugs into the code.

The actual solution, temporally before doing the refactor, is the new extra_params parameter.
This would be the GANDALF class, which contains all the global variables as class attributes.
'''

# # Train functions
@tf.function
def train_step(x, y, *y_discs, lambda_dict, extra_params):
    with tf.GradientTape() as tape_ae, tf.GradientTape(persistent=True) as tape_disc:
        # Obtaining latent space and autoencoder reconstruction
        z = extra_params.encoder(x, training=True) if extra_params.input_without_params else extra_params.encoder([x, y], training=True)
        output_ae = extra_params.decoder([z, y], training=True)

        # Autoencoder reconstruction error calculation
        loss_reconstruction = extra_params.loss_fn_reconstruction(x, output_ae)
        # Partial autoencoder error calculation
        loss_ae = loss_reconstruction

        # Obtaining the prediction of each discriminator and its error
        loss_disc_dict = dict()
        for (disc_name, discriminator), y_disc in zip(extra_params.disc_models_dict.items(), y_discs):
            # Obtaining discriminator predictions
            output_disc = discriminator(z if not extra_params.conv_disc else z[:,:,np.newaxis], training=True)

            # Discrimnator error calculation
            loss_disc_dict[disc_name] = extra_params.loss_fn_disc(y_disc, output_disc)

            # Calculation of the total error of the autoencoder applying the error of each discriminator to perform the adversarial training
            _weighted_lambda = lambda_dict[disc_name.replace('_discriminator','')] if extra_params.multi_disc else list(lambda_dict.values())[0]
            _loss_disc = -loss_disc_dict[disc_name]
            loss_ae += _weighted_lambda*_loss_disc

    # Calculation and application of gradients to modify the weights of the models
    for disc_name, discriminator in extra_params.disc_models_dict.items():
        grads_disc = tape_disc.gradient(loss_disc_dict[disc_name], discriminator.trainable_weights)
        extra_params.optimizer_disc.apply_gradients(zip(grads_disc, discriminator.trainable_weights))

    grads_ae = tape_ae.gradient(loss_ae, extra_params.autoencoder.trainable_weights)
    extra_params.optimizer_ae.apply_gradients(zip(grads_ae, extra_params.autoencoder.trainable_weights))

    return loss_reconstruction, loss_ae, loss_disc_dict

def train(dataset, model_id=None, epochs=100,
          search_ckpt=False, ckpt_step=10,
          saved_model_path='results/models',
          dynamic_learning=False, dl_no_change=None, dl_lambda_steps=None,
          verbose=1, extra_params=None):
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

    # If there is a checkpoint, the last one is recover
    if search_ckpt and extra_params.ckpt_manager.latest_checkpoint:
        extra_params.ckpt.restore(extra_params.ckpt_manager.latest_checkpoint)
        print('Checkpoint restored from {}'.format(extra_params.ckpt_manager.latest_checkpoint))
    else:
        print('No checkpoint found.')


    # Logs for tensorboard
    log_dir = os.path.join(extra_params.root_folder, "logs/", "fit/", model_id)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Metrics
    epoch_loss_reconstruction_mean = tf.keras.metrics.Mean('epoch_loss_reconstruction_mean', dtype=tf.float32)
    epoch_loss_ae_mean = tf.keras.metrics.Mean('epoch_loss_ae_mean', dtype=tf.float32)
    epoch_loss_disc_mean_dict = dict()
    for _key in extra_params.disc_models_dict.keys():
        epoch_loss_disc_mean_dict[_key] = tf.keras.metrics.Mean('epoch_loss_disc_{:}_mean'.format(_key), dtype=tf.float32)

    # Dynamic Learning
    _dl_marker = ''
    if dynamic_learning:
        _lambda_max = extra_params.lambda_dict
        # A dictionary is created for the current lambda values and another for the increment values
        _lambda = dict()
        _lambda_rate = dict()
        for cond_label in extra_params.cond_labels:
            _lambda[cond_label] = 0
            _lambda_rate[cond_label] = _lambda_max[cond_label]/dl_lambda_steps

        _epochs_no_change = dl_no_change
        _current_epochs_no_change = 0
        _best_ae_loss = sys.float_info.max
    else:
        _lambda = extra_params.lambda_dict
    # ----------------

    # Epochs
    for _ in range(epochs):
        # Checkpoint epoch value is increased
        extra_params.ckpt.epoch.assign_add(1)

        # The real epoch is obtained by checking the checkpoint
        real_epoch = int(extra_params.ckpt.epoch)

        if verbose != 0:
            print("\nEpoch", real_epoch, end='\r')

        # Error metrics are reset
        epoch_loss_ae_mean.reset_state()
        for _key in extra_params.disc_models_dict.keys():
            epoch_loss_disc_mean_dict[_key].reset_state()
        epoch_loss_reconstruction_mean.reset_state()

        # Unimproved iteration marker is created (dynamic_learning)
        if dynamic_learning:
            _dl_marker = '*'*_current_epochs_no_change

        time_start = time.time()
        # Batches
        for (x_batch_train, y_batch_train, *y_batch_train_encoded_list) in dataset:

            # Training
            loss_reconstruction, loss_ae, loss_disc_dict = train_step(x_batch_train,
                                                                      y_batch_train,
                                                                      *y_batch_train_encoded_list if extra_params.discretize else y_batch_train,
                                                                      lambda_dict=_lambda,
                                                                      extra_params=extra_params)

            # Error metrics are updated
            epoch_loss_reconstruction_mean.update_state(loss_reconstruction)
            epoch_loss_ae_mean.update_state(loss_ae)
            verbose_loss_string = str()
            for _key in extra_params.disc_models_dict.keys():
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
                for cond_label in extra_params.cond_labels:
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
            ckpt_save_path = extra_params.ckpt_manager.save(checkpoint_number=real_epoch)
            verbose and print('\nSaving checkpoint for epoch {} at {}'. format(real_epoch, os.path.abspath(ckpt_save_path)))

        # Log epoch metrics
        with summary_writer.as_default():
            tf.summary.scalar('reconstruction_loss', epoch_loss_reconstruction_mean.result(), real_epoch)
            tf.summary.scalar('autoencoder_loss', epoch_loss_ae_mean.result(), real_epoch)
            for _key in extra_params.disc_models_dict.keys():
                tf.summary.scalar(_key+'_loss', epoch_loss_disc_mean_dict[_key].result(), real_epoch)

    # At the end, a last checkpoint is created (if it has not already done in the last epoch)
    if real_epoch % ckpt_step != 0:
        ckpt_save_path = extra_params.ckpt_manager.save(checkpoint_number=real_epoch)
        verbose and print('\nSaving checkpoint for epoch {} at {}'.format(real_epoch, os.path.abspath(ckpt_save_path)))

    # At the end, the model is saved
    if saved_model_path is not None:
        _model_folder_path = os.path.normpath(os.path.join(extra_params.root_folder, saved_model_path, model_id))
        # Create folder if doesn't exist
        if not os.path.exists(_model_folder_path):
            print('\nCreating model folder in \'{:}\''.format(os.path.abspath(_model_folder_path)))
            os.makedirs(_model_folder_path)
        ae_model_save_path = os.path.abspath(os.path.join(_model_folder_path, "autoencoder.keras"))
        extra_params.autoencoder.save(ae_model_save_path)
        # All discriminator are saved
        for _name, _discriminator in extra_params.disc_models_dict.items():
            _disc_model_save_path = os.path.join(_model_folder_path, _name + '.keras')
            _discriminator.save(_disc_model_save_path)
        verbose and print('\nSaving models at {}'.format(os.path.abspath(_model_folder_path)), end='\n\n')
# ---

def eval_training(dataset_test, verbose=1, extra_params=None):
    '''Evaluation using the test dataset

    Parameters
    ----------
    dataset_test : Tensorflow.Dataset
        Test dataset
    verbose : bool, default True
        Verbosity mode
    '''
    verbose and print('\nTest results')
    verbose and print('------------')

    # Metric definitions
    _loss_reconstruction_metric = tf.keras.metrics.Mean('loss_reconstruction', dtype=tf.float32)
    _loss_ae_metric = tf.keras.metrics.Mean('loss_ae', dtype=tf.float32)
    _loss_disc_metric_dict = dict()
    for _key in extra_params.disc_models_dict.keys():
        _loss_disc_metric_dict[_key] = tf.keras.metrics.Mean('loss_disc_{:}'.format(_key), dtype=tf.float32)

    # Batch test error calculation
    for x_batch_test, y_batch_test, *y_batch_test_encoded_list in dataset_test:
        loss_reconstruction, loss_ae, loss_disc_dict = test_step(x_batch_test,
                                                                 y_batch_test,
                                                                 *y_batch_test_encoded_list if extra_params.discretize else y_batch_test,
                                                                 lambda_dict=extra_params.lambda_dict, extra_params=extra_params)
        # Error metrics are updated
        _loss_reconstruction_metric.update_state(loss_reconstruction)
        _loss_ae_metric.update_state(loss_ae)
        for _key in extra_params.disc_models_dict.keys():
            _loss_disc_metric_dict[_key].update_state(loss_disc_dict[_key])


    # Final results
    verbose and print('\nReconstruction loss: {:f}'.format(_loss_reconstruction_metric.result()))

    _loss_disc_mean = 0
    for _key in extra_params.disc_models_dict.keys():
        _loss_disc_mean += _loss_disc_metric_dict[_key].result()
        verbose and print('{:} loss: {:f}'.format(_key, _loss_disc_metric_dict[_key].result()))
    _loss_disc_mean /= len(extra_params.disc_models_dict)
    verbose and print('Autoencoder loss: {:f}'.format(_loss_ae_metric.result()))

    return _loss_ae_metric.result(), _loss_reconstruction_metric.result(), _loss_disc_mean
# ---

# # Test functions
def test_step(x, y, *y_discs, lambda_dict, extra_params):
    '''Get model losses'''

    # Obtaining latent space and autoencoder reconstruction
    z = extra_params.encoder(x, training=False) if extra_params.input_without_params else extra_params.encoder([x, y], training=False)
    output_ae = extra_params.decoder([z, y], training=False)

    # Autoencoder reconstruction error calculation
    loss_reconstruction = extra_params.loss_fn_reconstruction(x, output_ae)
    # Partial autoencoder error calculation
    loss_ae = loss_reconstruction

    # Obtaining the prediction of each discriminator and its error
    loss_disc_dict = dict()
    for (disc_name, discriminator), y_disc in zip(extra_params.disc_models_dict.items(), y_discs):
        # Obtaining discriminator predictions
        output_disc = discriminator(z if not extra_params.conv_disc else z[:,:,np.newaxis], training=False)

        # Discrimnator error calculation
        loss_disc_dict[disc_name] = extra_params.loss_fn_disc(y_disc, output_disc)

        # Calculation of the total error of the autoencoder applying the error of each discriminator to perform the adversarial training
        _weighted_lambda = lambda_dict[disc_name.replace('_discriminator','')] if extra_params.multi_disc else list(lambda_dict.values())[0]
        _loss_disc = -loss_disc_dict[disc_name]
        loss_ae += _weighted_lambda*_loss_disc

    return loss_reconstruction, loss_ae, loss_disc_dict
# ---