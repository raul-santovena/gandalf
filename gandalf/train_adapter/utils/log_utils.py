import os
from inspect import signature

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

    file_path = os.path.normpath(os.path.join(root_folder, file_path))

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