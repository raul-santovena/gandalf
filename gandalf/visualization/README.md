# Visualization tool
This tool allows you to visualize an autoencoder, showing the input data, the latent space and the reconstruction of the data. It also allows interactively modifying the latent space and generating new artificial examples. Test data and tensorflow model are required.

## Configuration
After creating and training a model with `train_cli` , an unique `id` is generated to identify it. If the training tool has not been modified, to display a model it is enough to update the value of the attribute `model_id` in [configuration.json](configuration.json).

The visualization tool also offers the possibility of exporting data generated with the autoencoder, you can configure the folder where you want to save this data by modifying the `export_folder` attribute.

## Running
To run the server, execute the command (example from the root of the project):

    bokeh serve --show gandalf/visualization

By default, a bokeh server will be started on port 5006 of your machine. You can access the tool using a browser and visiting the url http://localhost:5006/visualization.

More info about Bokeh servers in the [official site](https://docs.bokeh.org/en/latest/docs/user_guide/server.html).
