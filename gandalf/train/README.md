# Train module
This module allows you to define, train and test disentangling models. An example data set is available in train/data/sample to test its operation.

## Defining and training models
To create and train a model, use the command line tool train_cli.py. Through this tool you can configure the different parameters of the model and its training, as well as parameters related to the data to be used and their format. To see a description of each of them, use the --help flag as follow:
```
> python .\gandalf\train\train_cli.py --help
usage: train_cli [-h] [--survey {sample}]
                 [--survey_data_path SURVEY_DATA_PATH]
                 [--params PARAMS [PARAMS ...]]
                 [--cond_params COND_PARAMS [COND_PARAMS ...]]
                 [--training_id TRAINING_ID]
                 [--get_model_settings GET_MODEL_SETTINGS] [--normalize]     
                 [--shuffle] [--discretize] [--nbins NBINS]
                 [--batch_size BATCH_SIZE] [--seed SEED]
                 [--input_only_spectra] [--latent_size LATENT_SIZE]
                 [--encoder_hidden_layer_sizes [N [N ...]]]
                 [--decoder_hidden_layer_sizes [N [N ...]]]
                 [--discriminator_hidden_layer_sizes [N [N ...]]]
                 [--batch_normalization] [--convolutional_discriminator]     
                 [--multi_disc] [--summary_models] [--epochs EPOCHS]
                 [--lambda_values LAMBDA_VALUES [LAMBDA_VALUES ...]]
                 [--dynamic_learning NO_CHANGE STEPS]
                 [--discriminator_learning_rate DISCRIMINATOR_LEARNING_RATE] 
                 [--autoencoder_learning_rate AUTOENCODER_LEARNING_RATE] [-v]

Define and train disentangling models

optional arguments:
  -h, --help            show this help message and exit
  --survey {sample}     data source
  --survey_data_path SURVEY_DATA_PATH
                        Path where the data survey is located
  --params PARAMS [PARAMS ...]
                        parameters contained in the data
  --cond_params COND_PARAMS [COND_PARAMS ...]
                        parameters that will be decomposed
  --training_id TRAINING_ID, --model_id TRAINING_ID
                        model/training id to load a specific training,
                        including data and models. If this parameter is
                        passed, all parameters except the training ones are
                        ignored (currently, is the user who has to pass the
                        same parameters)
  --get_model_settings GET_MODEL_SETTINGS
                        return the command line instruction to continue the
                        training of the specified id

Data parameters:
  --normalize           Normalize X
  --shuffle             shuffle data before the dataset is created
  --discretize          discretizes the conditional parameters into bins for
                        training
  --nbins NBINS         number of bins
  --batch_size BATCH_SIZE
                        batch size
  --seed SEED           seed to replicate the results

Models parameters:
  --input_only_spectra  the conditional parameters are only passed to latent
                        space
  --latent_size LATENT_SIZE
                        size of the latent space in the autoencoder
  --encoder_hidden_layer_sizes [N [N ...]]
                        The ith element represents the number of neurons in
                        the ith hidden layer
  --decoder_hidden_layer_sizes [N [N ...]]
                        The ith element represents the number of neurons in
                        the ith hidden layer
  --discriminator_hidden_layer_sizes [N [N ...]]
                        The ith element represents the number of neurons in
                        the ith hidden layer
  --batch_normalization, --batch_norm
                        Add batch normalization between layers to both
                        autoencoder and discriminators
  --convolutional_discriminator, --conv_disc
                        make a convolutional discriminator instead of regular
                        ann layers
  --multi_disc, --multi_discriminator
                        create an architecture with a discriminator per
                        conditional parameter
  --summary_models, --summary
                        print the summary of the models

Training parameters:
  --epochs EPOCHS       Number of epoch during training
  --lambda_values LAMBDA_VALUES [LAMBDA_VALUES ...]
                        Lambda values to control the disentanglement of each
                        parameter
  --dynamic_learning NO_CHANGE STEPS, --dynamic_lambda NO_CHANGE STEPS
                        use a dynamic lambda value during training
  --discriminator_learning_rate DISCRIMINATOR_LEARNING_RATE, --disc_lr DISCRIMINATOR_LEARNING_RATE
                        Learning rate of the discriminator
  --autoencoder_learning_rate AUTOENCODER_LEARNING_RATE, --ae_lr AUTOENCODER_LEARNING_RATE
                        Learning rate of the autoencoder
  -v, --verbose         verbose
```
During the course and completion of training, several useful folders and files are generated such as a log record to load with tensorboard, training checkpoints, a results record with model parameters, training and performance metrics. 

For each new model, an id is generated that can later be used for visualization. More details in [gandalf/gandalf/visualization/README.md](../visualization/README.md)

## Testing disentangling
Once a model is created and trained, various tests can be performed to check its effectiveness. This is done with the command line tool test_cli.py. To see a description of each of the options, use the --help flag:
```
> python .\gandalf\train\test_cli.py --help 
usage: test_cli [-h] [--model_names name [name ...]] [--seed SEED] [-v]
                [--all] [--tsne_comparison]
                [--conditional_parameters_analysis]
                [--no_conditional_parameters_analysis] [--models_comparison]
                [--repetitions REPETITIONS] [--mlp_max_iter MLP_MAX_ITER]
                [--x_hidden_layer_sizes [N [N ...]]]
                [--z_hidden_layer_sizes [N [N ...]]] [--latex_format]
                [--export_graph]
                id [id ...]

Test disentanglement of the models through several test

positional arguments:
  id                    model ids to load the data generated during training

optional arguments:
  -h, --help            show this help message and exit
  --model_names name [name ...]
                        model names to be visualized with the results. The
                        length must match the number of ids
  --seed SEED           Value to replicate the results
  -v, --verbose         verbosity

Feasible tests:
  --all                 run all tests. If this parameter is passed, the remain
                        parameters are ignored
  --tsne_comparison     comparison of spectra and latent space using the tsne
                        algorithm
  --conditional_parameters_analysis
                        analyze the result of predict the conditional
                        parameters from the spectra vs from the latent space
  --no_conditional_parameters_analysis
                        analyze the result of predict no conditional
                        parameters from the spectra vs from the latent space
  --models_comparison   compare a list of models

Test parameters:
  --repetitions REPETITIONS
                        Repetition of each training in cond and non-cond
                        parameter analysis
  --mlp_max_iter MLP_MAX_ITER
                        Maximun number of iterations of used MLPs. Default 600
                        iterations
  --x_hidden_layer_sizes [N [N ...]]
                        The ith element represents the number of neurons in
                        the ith hidden layer
  --z_hidden_layer_sizes [N [N ...]]
                        The ith element represents the number of neurons in
                        the ith hidden layer
  --latex_format        generate results in latex table format

Tsne parameters:
  --export_graph        Export a png graph with the comparisons
```