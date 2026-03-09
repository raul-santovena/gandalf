from bokeh.io import curdoc
import sys
import os

# My modules
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.normpath('../../'))) # To support the use of the tool without packaging
from gandalf.visualization.app import GandalfVisualiser

if len(sys.argv) != 2:
    print('Usage: bokeh serve gandalf/visualization --args <config_file>')
    sys.exit(1)

config_file = sys.argv[1]

gandalf_visualiser = GandalfVisualiser(config_file)

curdoc().add_root(gandalf_visualiser.get_layout())