import os
import sys
import numpy as np
from sklearn.manifold import TSNE

from bokeh.models import ColorBar, ColumnDataSource
from bokeh.plotting import figure, show
from bokeh.transform import linear_cmap
from bokeh.palettes import RdYlBu9
from bokeh.layouts import row, column
from bokeh.io import export_png

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# My modules
sys.path.append("../") # To support the use of the tool without packaging
import gandalf.train.utils as my_utils

# Constants
WIDTH = 100

def _tsne_bokeh(embedded_x, embedded_z, y_cond, cond_name_labels, id, root_folder, name=None, export_graph=False, verbose=1):
    verbose and print('Generating graph... ', end='')

    _colors = RdYlBu9[::-1]

    rows = list()

    for i, cond_name_label in enumerate(cond_name_labels):
        _y = y_cond[:, i]

        source = ColumnDataSource(dict(z_x=embedded_z[:,0],
                                    z_y=embedded_z[:,1],
                                    x_x=embedded_x[:,0],
                                    x_y=embedded_x[:,1],
                                    cond_label=_y))

        # use the field name of the column source
        cmap = linear_cmap(field_name='cond_label', palette=_colors, low=min(_y), high=max(_y))


        p1 = figure(width=900, height=900, title="")
        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.axis.visible = False
        p1.scatter(x='x_x', y='x_y', color=cmap, size=15, source=source)

        p2 = figure(width=900, height=900, title="")
        p2.toolbar.logo = None
        p2.toolbar_location = None
        p2.axis.visible = False
        p2.scatter(x='z_x', y='z_y', color=cmap, size=15, source=source)

        # pass the mapper's transform to the colorbar
        color_bar = ColorBar(color_mapper=cmap['transform'], width=10, title=cond_name_label)

        p1.add_layout(color_bar, 'right')
        p2.add_layout(color_bar, 'right')

        rows.append(row(p1,p2))

    full_layout = column(*rows)
    show(full_layout)

    if export_graph:
        filename = 'tsne_comparison_{:}_{:}.png'.format(id, '_'.join(cond_name_labels))
        filename_path = os.path.join(root_folder, filename)
        export_png(full_layout, filename=filename_path)
        verbose and print('and saving it at {:}'.format(os.path.abspath(filename_path)), end='\n\n')
    else:
        verbose and print('Done', end='\n\n')

def _tsne_matplotlib(embedded_x, embedded_z, y_cond, cond_name_labels, id, root_folder, name=None, export_graph=False, verbose=1):
    verbose and print('Generating graph... ', end='')

    # Define the color map for visualization
    cmap = plt.get_cmap('RdYlBu', 9)

    rows = len(cond_name_labels)

    fig, axes = plt.subplots(rows, 2, figsize=(12, 6 * rows))

    # If we only have one row, axes will be a 1D array
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, cond_name_label in enumerate(cond_name_labels):
        _y = y_cond[:, i]

        # Normalize the color map based on the condition values
        norm = Normalize(vmin=min(_y), vmax=max(_y))

        # Scatter plot for embedded_x
        ax1 = axes[i, 0]
        scatter1 = ax1.scatter(embedded_x[:, 0], embedded_x[:, 1], c=_y, cmap=cmap, norm=norm, s=15)
        ax1.set_title(f"{cond_name_label} (x space)")
        ax1.axis('off')  # Hide axes

        # Scatter plot for embedded_z
        ax2 = axes[i, 1]
        scatter2 = ax2.scatter(embedded_z[:, 0], embedded_z[:, 1], c=_y, cmap=cmap, norm=norm, s=15)
        ax2.set_title(f"{cond_name_label} (z space)")
        ax2.axis('off')  # Hide axes

        # Add colorbars for each subplot (increase aspect to make it smaller)
        fig.colorbar(scatter1, ax=ax1, orientation='vertical', label=cond_name_label, aspect=35)
        fig.colorbar(scatter2, ax=ax2, orientation='vertical', label=cond_name_label, aspect=35)

    # Adjust layout
    plt.tight_layout()

    # Show or export the plot
    if export_graph:
        filename = 'tsne_comparison_{:}_{:}.png'.format(id, '_'.join(cond_name_labels))
        filename_path = os.path.join(root_folder, filename)
        plt.savefig(filename_path, dpi=300)
        verbose and print('and saving it at {:}'.format(os.path.abspath(filename_path)), end='\n\n')
    else:
        plt.show()
        verbose and print('Done', end='\n\n')

def tsne_comparison(id_, root_folder, name=None, random_state=1, cache='tests/tsne',
                    export_graph=False, backend='bokeh',
                    verbose=0):
    '''Compares the distribution of the original data and the latent space of the autoencoder, based on the
    parameters, and using the tsne algorithm.

    Parameters
    ----------
    id_ : str
        Identifier from which the trained model and the data corresponding to it are obtained.
    root_folder : str
        Path to the root folder where the data is located
    name: str, default None
        Name of the model to put as the title of the graph, if not specified, the id_ is used
    random_state: int, default 1
        Seed for the tsne algorithm, in order to be able to replicate the results
    cache: str, default 'tests/tsne'
        If it is not None, look in the indicated path for the folder named "id_", the files "x_<random_state>.npy" and
        "z_<random_state>.npy". If they exist, it loads them instead of running tsne. If it does not exist, run tsne and
        save the results.
    export_graph: bool, default False
        Export the graph to disk in png format
    backend: str, default 'bokeh'
        Backend to use for plotting the graph. Options: 'bokeh' or 'matplotlib'
    verbose : {0, 1, 2}, default 0
        Verbose, if 1, print messages, if 2 shows t-SNE progress
    '''

    print('-'*WIDTH)
    print('t-SNE comparison'.center(WIDTH))
    print('-'*WIDTH)

    show_progress = verbose == 2

    verbose and print('Starting tsne comparison...')
    # Load data
    data_dict = my_utils.load_data(id_=id_, base_dir=root_folder)

    # It checks if there is saved data related to the application of the tsne to this dataset with the same random_state
    saved_embedded_x_path = os.path.normpath(os.path.join(root_folder, cache, id_, 'x_{}.npy'.format(random_state)))
    saved_embedded_z_path = os.path.normpath(os.path.join(root_folder, cache, id_, 'z_{}.npy'.format(random_state)))

    ## X
    if not os.path.exists(saved_embedded_x_path): # If there is no previous data, the tsne is executed and the data is saved
        verbose and print('There is no cached data. Applying tsne to the original spectra... ', end='')
        tsne = TSNE(n_components=2, max_iter=1000, perplexity=30, learning_rate='auto', init='pca',
                    random_state=random_state, n_jobs=-1, verbose=show_progress)

        embedded_x = tsne.fit_transform(data_dict['X'])

        # Create folder if doesn't exist
        _folder_path = os.path.dirname(saved_embedded_x_path)
        if not os.path.exists(_folder_path):
            print('Folder \'{:}\' doesn\'t exist! Created.'.format(os.path.abspath(_folder_path)))
            os.makedirs(_folder_path)

        np.save(saved_embedded_x_path, embedded_x)
        verbose and print('Done. Data saved in {}'.format(os.path.abspath(saved_embedded_x_path)))

    else: # If previous data exists, it is loaded
        verbose and print('Located data saved in {}'.format(os.path.abspath(saved_embedded_x_path)))
        embedded_x = np.load(saved_embedded_x_path, allow_pickle=True)

    ## z
    if not os.path.exists(saved_embedded_z_path): # If there is no previous data, the tsne is executed and the data is saved
        verbose and print('There is no cached data. Applying tsne to the latent space... ', end='')
        tsne = TSNE(n_components=2, max_iter=1000, perplexity=30, learning_rate='auto', init='pca',
                    random_state=random_state, n_jobs=-1, verbose=show_progress)

        embedded_z = tsne.fit_transform(data_dict['z'])

        # Create folder if doesn't exist
        _folder_path = os.path.dirname(saved_embedded_z_path)
        if not os.path.exists(_folder_path):
            print('Folder \'{:}\' doesn\'t exist! Created.'.format(os.path.abspath(_folder_path)))
            os.makedirs(_folder_path)

        np.save(saved_embedded_z_path, embedded_z)
        verbose and print('Done. Data saved in {}'.format(os.path.abspath(saved_embedded_z_path)))
    else: # If previous data exists, it is loaded
        verbose and print('Located data saved in {}'.format(os.path.abspath(saved_embedded_z_path)))
        embedded_z = np.load(saved_embedded_z_path, allow_pickle=True)
    # -----------------------------------------------------------------------------------------------------------------------------


    # Plot
    if backend == 'bokeh':
        _tsne_bokeh(embedded_x=embedded_x, embedded_z=embedded_z,
                    y_cond=data_dict['cond_params'],
                    cond_name_labels=data_dict['cond_params_names'],
                    id=id_, root_folder=root_folder,
                    name=name if name is not None else id_,
                    export_graph=export_graph,
                    verbose=1)
    elif backend == 'matplotlib':
        _tsne_matplotlib(embedded_x=embedded_x, embedded_z=embedded_z,
                        y_cond=data_dict['cond_params'],
                        cond_name_labels=data_dict['cond_params_names'],
                        id=id_, root_folder=root_folder,
                        name=name if name is not None else id_,
                        export_graph=export_graph,
                        verbose=1)
    else:
        raise ValueError(f"Invalid backend: {backend}")