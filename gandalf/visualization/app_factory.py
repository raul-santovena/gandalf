from bokeh.io import show
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.normpath('../../'))) # To support the use of the tool without packaging
from gandalf.visualization.app import GandalfVisualiser

ICON_PATH = "static/favicon.ico"

class GandalfVisualiserFactory():
    """
    A factory class for creating a new GandalfVisualiser instance for each Bokeh session.

    This is necessary because Bokeh requires that all models (widgets, plots, etc.)
    be unique for each document session. If the same GandalfVisualiser instance were
    reused, Bokeh would attempt to attach already-initialized models to a new document,
    causing a RuntimeError.

    This issue is especially problematic in the `bokeh_server` function. If a user
    accesses the Bokeh URL twice, Bokeh will try to create a new document but reuse
    the same models from a previous session, leading to a crash.

    By creating a new GandalfVisualiser instance inside `_modify_doc`, this factory
    ensures that every session gets a fresh set of Bokeh objects, preventing conflicts.
    """
    def __init__(self, config_file):
        self.config_file = config_file

    def _modify_doc(self, doc):
        """
        Helper function to set up the layout for Bokeh visualization.
        """
        gandalf_visualiser = GandalfVisualiser(self.config_file)
        layout = gandalf_visualiser.get_layout()
        doc.add_root(layout)
        doc.title = "GANDALF Visualiser"

def bokeh_notebook(config_file, notebook_url="http://localhost:8888", port=5006):
    """
    Initializes and shows a Bokeh visualization with a given configuration file in a Jupyter Notebook.

    Args:
        config_file (str): Path to the configuration file for the visualization.

    Example:
        bokeh_visualization("visualization_config.yaml")
    """

    # Example: https://github.com/bokeh/bokeh/blob/2.4.1/examples/howto/server_embed/notebook_embed.ipynb

    gandalf_visualiser = GandalfVisualiserFactory(config_file)
    show(gandalf_visualiser._modify_doc, notebook_url=notebook_url, port=port)

def bokeh_server(config_file, port=5006):
    """
    Starts a Bokeh server to host the GANDALF visualization.

    Args:
        config_file (str): Path to the configuration file for the visualization.
        port (int, optional): Port on which to run the Bokeh server. Default is 5006.

    The server will be accessible at the URL specified by the environment variable `BOKEH_URL` if it exists (`<BOKEH_URL>:<port>/`).
    If `BOKEH_URL` is not set, the server will default to `http://localhost:<port>/`.
    """

    # Example: https://stackoverflow.com/questions/53217654/how-to-get-interactive-bokeh-in-jupyter-notebook

    gandalf_visualiser = GandalfVisualiserFactory(config_file) # Factory creates a new instance each time the user accesses the Bokeh URL
    app = Application(FunctionHandler(gandalf_visualiser._modify_doc))

    server = Server(app, port=port, ico_path=os.path.join(os.path.dirname(__file__), ICON_PATH))
    server.start()

    server_url = os.getenv('BOKEH_URL', 'http://localhost') # TODO: it is not the Bokeh URL, it is the server URL

    # If the server URL ends with 'proxy/' it should not add an extra ':'
    server_url_full = f"{server_url}/{port}" if server_url.endswith('proxy/') else f"{server_url}:{port}"

    print(f"Server running at {server_url_full}")
