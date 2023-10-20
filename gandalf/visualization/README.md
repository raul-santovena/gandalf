# Visualization module (demo with astronomical data)
This tool allows you to visualize an autoencoder, showing the input data, the latent space and the reconstruction of the data. It also allows interactively modifying the latent space and generating new artificial examples. Test data and tensorflow model are required.

## Setting up
This tool requires some libraries in order to run. To install them using conda, execute the command:

    conda install numpy pandas tensorflow bokeh
  
To install using pip, execute the command:

    pip install numpy pandas tensorflow bokeh

## Configuration.json
After creating and training a model, a unique ID is generated to identify it. If the training tool has not been modified, to display a model it is enough to update the value of the "model_id" attribute of the file [configuration.json](configuration.json).

## Running
To run the server, execute the command (example from the root of the project):

    bokeh serve --show gandalf/visualization

By default, a bokeh server will be started on port 5006 of your machine. You can access the tool using a browser and visiting the url http://localhost:5006/visualization