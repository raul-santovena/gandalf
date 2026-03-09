from abc import ABC, abstractmethod

# GANDALF Train Abstract Class (interface)
class GandalfTrainInterface(ABC):
    ''' Abstract class to define and train disentangling models using GANDALF '''

    def __init__(self, config):
        ''' Constructor for GANDALFTrain class

        Parameters
        ----------
        config : dict
            Dictionary with the configuration parameters for the training process (see schema in /schemas/config.json)
        '''
        self._config = config

        ## General arguments
        self.root_folder = None
        self.model_id = None

        ## Data arguments
        self.data_loader = None
        self.survey_data_path = None
        self.labels = None
        self.cond_labels = None
        self.normalize_spectra = None
        self.shuffle = None
        self.discretize = None
        self.nbins = None
        self.seed = None

        # Model
        ## Arguments
        self.encoder_hidden_layer_sizes = None
        self.decoder_hidden_layer_sizes = None
        self.latent_size = None
        self.input_without_params = None
        self.batch_norm = None
        self.disc_hidden_layer_sizes = None
        self.conv_disc = None
        self.multi_disc = None

        ## Models used during training
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.disc_models_dict = None

        # Training
        ## Arguments
        self.epochs = None
        self.batch_size = None
        self.lambda_values = None
        self.lr_ae = None
        self.lr_disc = None
        self.dl_no_change = None
        self.dl_lambda_steps = None

        # Save for results CSV
        self.loss_fn_reconstruction = None
        self.loss_fn_disc = None
        self.optimizer_disc = None
        self.optimizer_ae = None
        self.ckpt_manager = None
        self.ckpt = None
        self.lambda_dict = None
        self.dynamic_learning = None

    @classmethod
    @abstractmethod
    def from_config_file(cls, config_file, file_type='yaml'):
        '''Create a GANDALFTrain object from a configuration file

        Parameters
        ----------
        config_file : str
            Path to the configuration file
        file_type : {yaml}, default 'yaml'
            Type of configuration file

        Returns
        -------
        GANDALFTrain object
        '''
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    def create_datasets(self, verbose=0):
        '''Create a train and a test dataset for use with GANDALF.

        Parameters
        ----------
        verbose : {0, 1 o 2}, default 0
            Level of verbosity

        Returns
        --------
        train_dataset : tf.data.Dataset
            A tensorflow dataset with the training data
        test_dataset : tf.data.Dataset
            A tensorflow dataset with the testing data
        '''
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    def create_autoencoder(self, data_dim, summary_models=False):
        '''Create the autoencoder tensorflow model

        Parameters
        ----------
        data_dim : int
            Size of the one-dimensional array that represents the data
        summary_models : bool, default False
            If True, print the model summary
        '''
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    def create_discriminator(self, output_discriminator, summary_models=False):
        '''Create the tensorflow model of the discriminator

        Parameters
        ----------
        output_discriminator : int
            Network output array size
        summary_models : bool, default False
            If True, print the model summary
        '''
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    def train(self, dataset, search_ckpt=False, ckpt_step=10, saved_model_path='results/models', verbose=1):
        '''
        Perform model training.

        Parameters
        ----------
        dataset : tf.data.dataset
            Training dataset
        search_ckpt : bool, default False
            Searches if there is a checkpoint for the current training, and loads it
        ckpt_step : int, default 10
            Frequency (in epochs) to create checkpoints
        saved_model_path : str, default 'results/models'
            Saves the model at the end of training in the path specified by parameters, if it is None, the model is not saved
        verbose : {0, 1 o 2}, default 1
            Verbose mode. 0 does not show output on screen, 1 output updated by epoch, 2 output updated by batch step
        '''
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    def eval_training(self, dataset_test, verbose=1):
        '''Evaluation using the test dataset

        Parameters
        ----------
        dataset_test : Tensorflow.Dataset
            Test dataset
        verbose : bool, default True
            Verbosity mode

        Returns
        -------
        ae_loss : float
            Autoencoder loss (reconstruction + lambda * discriminator)
        reconstruction_loss : float
            Reconstruction loss
        disc_loss : float
            Discriminator loss
        '''
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    def update_results_csv(self, reconstruction_loss, disc_loss, ae_loss):
        '''Save the data of the training

        Parameters
        ----------
        reconstruction_loss : float
            Reconstruction loss
        disc_loss : float
            Discriminator loss
        ae_loss : float
            Autoencoder loss
        update_json : bool, default False
            If True, the results are updated in the JSON file
        '''
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    def save_visualization_data(self, dataset_train, dataset_test, update_json=False):
        '''Save the data of the training

        Parameters
        ----------
        dataset_train : tf.data.dataset
            Training dataset
        dataset_test : tf.data.dataset
            Test dataset
        update_json : bool, default False
            If True, the results are updated in the JSON file
        '''
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    def show_arguments(self):
        '''Show the arguments of the training'''
        raise NotImplementedError("Method not implemented!")
