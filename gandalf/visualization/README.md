# Visualization tool

This tool allows you to visualize an autoencoder, showing the input data, the latent space and the reconstruction of the data. It also allows interactively modifying the latent space and generating new artificial examples. Test data and tensorflow model are required.

It can be used in two ways:

 - From the repository with `bokeh serve --show gandalf/visualisation --args gandalf/visualization/visualization_config.yaml`.
 - In Jupyter Lab, importing the `bokeh_visualization` or the `bokeh_server` functions (see sample notebooks).

## Configuration
After creating and training a model with `train_cli` or `GandalfTrain`, an unique `id` is generated to identify it. The visualizer reads this `id` and other parameters from a YAML configuration file. An example is shown below:

````yaml
model:
  model_name: "CAE"
  model_id: "id"
  base_dir: "root_folder"
  export_folder: "generated_samples"
  X_filename: "X_test.npy"
  params_filename: "params_test.npy"
  cond_params_filename: "cond_params_test.npy"
  z_filename: "z_test.npy"
  decoded_filename: "decoded_test.npy"
  ids_filename: "ids_test.npy"
  axis_labels_filename: "axis_labels.npy"
  params_names_filename: "params_names.npy"
  cond_params_names_filename: "cond_params_names.npy"
  X_scaler_filename: "X_scaler.pkl"
  param_scaler_filename: "param_scaler.pkl"

info:
  version: 1.0

````

The ``base_dir`` field refers to the folder in which the different experiments are stored. This must be the same as the ``root_folder`` parameter in ``train_cli`` or ``GandalfTrain``.

The visualization tool also offers the possibility of exporting data generated with the autoencoder, you can configure the folder where you want to save this data by modifying the `export_folder` attribute.

The other parameters don't need to be changed. They may be deleted in the future.

## Running

### Bokeh serve

To run the server, execute the command (example from the root of the project):

```
bokeh serve --show gandalf/visualization --args gandalf/visualization/visualization_config.yaml
```

By default, a bokeh server will be started on port 5006 of your machine. You can access the tool using a browser and visiting the url http://localhost:5006/visualization.

More info about Bokeh servers in the [official site](https://docs.bokeh.org/en/latest/docs/user_guide/server.html).

### Bokeh Server class

A server can be run using the `Server` class from Bokeh in Python, in a script or in any notebook. To simplify the iteration with the `Server` class, the `bokeh_server` function  will create and start the server, giving us the URL to access the application.

```python
bokeh_server(config_file, port=5006)
```

The configuration file used is the same as that defined previously. The default port is 5006, so the application will run at http://localhost:5006/.

An example can be found in `notebooks/bokeh-gandalf-server.ipynb`.

More info about Bokeh Server class in the [official documentation](https://docs.bokeh.org/en/3.6.2/docs/user_guide/server/library.html#ug-server-library).

### Bokeh embeded notebook

A Bokeh application can be embedded in a Jupyter Notebook. The ``bokeh_notebook`` function is available for this purpose. You will need **Jupyter Lab** to work with it, or run the available Dockerfile, which will open a Jupyter environment.

```python
bokeh_notebook(config_file, notebook_url="http://localhost:8888", port=5006)
```

The configuration file used is the same as the one defined previously. The notebook URL must be the same as the Jupyter lab URL, if we run it with the Dockerfile it will be http://localhost:8888. Finally, the port to run the Bokeh application, by default 5006. This needs to be exposed in the Dockerfile.

An example can be found in `notebooks/bokeh-gandalf-embeded.ipynb`.

More info about Bokeh servers embedded in a Notebook in the [official example](https://github.com/bokeh/bokeh/blob/2.4.1/examples/howto/server_embed/notebook_embed.ipynb).
