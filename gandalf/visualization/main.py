from glob import glob
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from bokeh.models import ColumnDataSource, TapTool, Slider, Select, Button, TextInput, Label
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.events import Pan
import json
import pickle

logging.basicConfig(level=logging.INFO)

def load_configuration_data():
    '''Load the file configuration.json as a dictionary'''

    with open(os.path.join(FILE_PATH, "configuration.json"), 'r') as jsonfile:
        _config_data = json.load(jsonfile)

    # Obtaining the 'model' object from the json
    _config_dict = _config_data['model']
    
    return _config_dict


def load_scalers():
    '''Load sklearn scalers to reverse normalized data'''

    _config_dict = load_configuration_data()

    _base_dir = os.path.normpath(_config_dict['base_dir'])
    _model_id = _config_dict['model_id']
    _X_scaler_filename = _config_dict['X_scaler_filename']
    _y_scaler_filename = _config_dict['param_scaler_filename']

    _full_data_path = os.path.join(FILE_PATH, _base_dir, 'data', _model_id)

    # Load scalers (QuantileTransformers)
    _param_scaler = pickle.load(open(os.path.join(_full_data_path, _y_scaler_filename), 'rb'))
    _X_scaler = pickle.load(open(os.path.join(_full_data_path, _X_scaler_filename), 'rb'))

    return _X_scaler, _param_scaler  


def load_data():
    '''Load the necessary data for the autoencoder'''

    _config_dict = load_configuration_data()
    
    # Building full data path
    _base_dir = os.path.normpath(_config_dict['base_dir'])
    _model_id = _config_dict['model_id']
    _full_data_path = os.path.join(FILE_PATH, _base_dir, 'data', _model_id)

    logging.info('Loading data from {}'.format(_full_data_path))

    # Filenames
    _X_filename = _config_dict['X_filename']
    _params_filename = _config_dict['params_filename']
    _cond_params_filename =_config_dict['cond_params_filename']
    _z_filename = _config_dict['z_filename']
    _decoded_filename = _config_dict['decoded_filename']
    _ids_filename = _config_dict['ids_filename']
    _axis_labels_filename = _config_dict['axis_labels_filename']
    _params_names_filename = _config_dict['params_names_filename']
    _cond_params_names_filename = _config_dict['cond_params_names_filename']

    # Data
    _X = np.load(os.path.join(_full_data_path, _X_filename), allow_pickle=True)
    _z = np.load(os.path.join(_full_data_path, _z_filename), allow_pickle=True)
    _decoded = np.load(os.path.join(_full_data_path, _decoded_filename), allow_pickle=True)
    _ids = np.load(os.path.join(_full_data_path, _ids_filename), allow_pickle=True)
    _axis_labels = np.load(os.path.join(_full_data_path, _axis_labels_filename), allow_pickle=True)
    _cond_params = np.load(os.path.join(_full_data_path, _cond_params_filename), allow_pickle=True)
    _params = np.load(os.path.join(_full_data_path, _params_filename), allow_pickle=True)

    _df = pd.DataFrame.from_dict(data={'ids':_ids.tolist(), 'X':_X.tolist(), 
                                 'z':_z.tolist(), 'decoded':_decoded.tolist(),
                                 })

    # Parameters names
    _params_names = np.load(os.path.join(_full_data_path, _params_names_filename), allow_pickle=True)
    _cond_params_names = np.load(os.path.join(_full_data_path, _cond_params_names_filename), allow_pickle=True)

    # We get each unscaled parameter and store them in an individual column
    _params_names = [_param_name.lower() for _param_name in _params_names]

    for _param_name in _params_names:
        _df['original_'+_param_name] = _params[:, np.isin(_params_names, [_param_name])].flatten().astype(str)

    # Idem with conditional parameters
    _cond_params_names = [_cond_param_name.lower() for _cond_param_name in _cond_params_names]

    for _cond_param_name in _cond_params_names:
        _df[_cond_param_name] = _cond_params[:, np.isin(_cond_params_names, [_cond_param_name])].flatten().astype(str)
   

    return _df, _axis_labels, _params_names, _cond_params_names


def load_model():
    '''Load the autoencoder model'''
    _config_dict = load_configuration_data()

    _base_dir = os.path.normpath(_config_dict['base_dir'])
    _model_id = _config_dict['model_id']
    _full_model_path = os.path.join(FILE_PATH, _base_dir, 'models', _model_id, 'autoencoder')

    return tf.keras.models.load_model(filepath=_full_model_path)

def gcd (a,b):
    if (b == 0):
        return a
    else:
         return gcd (b, round(a % b, 2))

def gcd_list(l):
    result = l[0]

    for item in l:
        result = gcd(result, item)
        
    return result


# VARIABLES #
FILE_PATH = os.path.dirname(__file__)

model = load_model()
is_conditional = True

df, axis_labels, params_names, cond_params_names = load_data()

X_scaler, param_scaler = load_scalers() 

# A dictionary is created for the original parameters with their unique values
unique_original_dict = dict()
for _param_name in params_names:
    unique_original_dict[_param_name] = np.unique(df['original_' + _param_name]).tolist()

unique_ids = np.unique(df.ids).tolist()

resolution = 110

id_selected = '-'
# A dictionary is created for the current values of each select of the astrophysical parameters
params_selected = dict()
for _param_name in params_names:
    params_selected[_param_name] = '-'

index_ = 0

## COLUMNDATASOURCES ##
def get_empty_X_source():
    return {'axis_labels': np.array([]),
            'X': np.array([]),
            'decoded_X': np.array([]),
            'generated_X': np.array([]),
            'comparison_X': np.array([])}

def get_empty_z_source():
    return {'z': np.array([]),
            'z_axis_labels': np.array([])}

# Creation of the ColumnDataSource (as all fields must have the same length, a ColumnDataSource is needed for the latent space and another for the spectra)
X_source = ColumnDataSource(data=get_empty_X_source())

z_axis_labels = list(range(len(df.z.iloc[index_])))

z_source = ColumnDataSource(data=get_empty_z_source())

########################

# FIGURES & WIDGETS #

## Widgets
### TextInput
search_text_input = TextInput(title='Search by id:', value='', placeholder='identifier')
### Selects
ids_select = Select(title='Identifier', value=None, options=['-'] + unique_ids)
#### Selects for parameters, we create a dictionary that contains the different selects of astrophysical parameters
selects_y_dict = dict() # Select dict
for _param_name in params_names:
    _select = Select(title=_param_name, value='', disabled=True)
    selects_y_dict[_param_name] = _select

### Reset button
reset_button = Button(label='Reset', button_type='danger', 
                      width=150, height=30, sizing_mode='fixed')

### Sliders (https://docs.bokeh.org/en/latest/docs/user_guide/interaction/widgets.html#slider)
z_slider = Slider(start=np.min(np.min(df.z.values)), end=np.max(np.max(df.z.values)), value=0, step=0.05, title='Selected Neuron Value',
                  width=150, height=50, sizing_mode='fixed') 

#### Sliders for conditional parameters
sliders_cond_params_dict = dict() # Slider dict

for _cond_param_name in cond_params_names:

    _unique_values = [float(_value) for _value in unique_original_dict[_cond_param_name]] # se convierte str a float para inicializar los valores del slider

    _slider = Slider(title=_cond_param_name, start=_unique_values[0], end=_unique_values[-1], value=np.median(_unique_values),
                     step=gcd_list(_unique_values),
                     width=150, height=50, sizing_mode='fixed')

    sliders_cond_params_dict[_cond_param_name] = _slider

### Export button
export_button = Button(label='Generate sample', button_type='success', align='end',
                       width=150, height=30, sizing_mode='fixed', margin=(5,40,5,5), disabled=True)

# Figures
## Plot original spectrum
_X = np.array(df.X.values[0])
y_range_min = np.min(_X) if X_scaler is None else np.min(X_scaler.inverse_transform(_X.reshape(1, -1)))
y_range_max = np.max(_X) if X_scaler is None else np.max(X_scaler.inverse_transform(_X.reshape(1, -1)))
f1 = figure(width=18, height=3, sizing_mode='stretch_both', tools=['pan', 'box_zoom', 'xwheel_zoom', 'reset'],
            y_range=(y_range_min, y_range_max), output_backend='webgl')
f1.line(x='axis_labels', y='X', source=X_source, line_color='#ECA400', line_width=2)
f1.toolbar.autohide=True


## Plot latent
f2 = figure(width=18, height=3, sizing_mode='stretch_both', tools='', output_backend='webgl')
f2.line(x='z_axis_labels', y ='z', source=z_source)
f2.circle(x='z_axis_labels', y ='z', source=z_source, size=8, line_color='navy', fill_color='orange', fill_alpha=1)

f2.add_tools(TapTool())
f2.toolbar.autohide=True


## Plot decoded spectrum
f3 = figure(width=18, height=3, x_range=f1.x_range, y_range=f1.y_range,
            sizing_mode='stretch_both', output_backend='webgl')
f3.line(x='axis_labels', y='comparison_X', source=X_source, line_color='#D5573B', line_width=2,
        legend_label='original')
f3.line(x='axis_labels', y='generated_X', source=X_source, line_color='#61988E', line_width=2,
        legend_label='generated')
f3.line(x='axis_labels', y='decoded_X', source=X_source, line_color='#27476E', line_width=2,
        legend_label='decoded')
        
comparison_label = Label(x=axis_labels[0], y=-1, x_units='data', y_units='data', 
                         text='', text_color='red',
                         border_line_color='white', border_line_width=5,
                         background_fill_color='white',
                         text_font_size='18px')
f3.add_layout(comparison_label)
f3.toolbar.autohide=True

## Common figure properties
for f in [f1, f2, f3]:
    f.toolbar.autohide=True
    f.background_fill_color = (250, 250, 250)
####################

# CALLBACKS #
automatic_change = False

def modify_neuron_callback(attr, old, new):
    '''Replaces the value of the selected neurons in the latent space with the value selected in the slider'''
    logging.info('Slider value: {}'.format(new))
    logging.info('Indices: {}'.format(z_source.selected.indices))
    global automatic_change

    if X_source.data['X'].size == 0: # If the arrays are empty, nothing is done (just check one)
        return

    if automatic_change:
        automatic_change = False
        return

    _z = z_source.data['z']
    _z_axis_labels = z_source.data['z_axis_labels']

    _z[z_source.selected.indices] = new

    z_source.data = dict(z = _z, z_axis_labels = _z_axis_labels)

    if is_conditional:
        # We obtain the values of the different conditional sliders
        _slider_values = list()
        for _label in cond_params_names:
            _slider_values.append(sliders_cond_params_dict[_label].value)

        # We apply the scaler
        _cond_values = param_scaler.transform([_slider_values])

        # Obtaining the modified spectrum
        _generated_X = model.get_layer('decoder')([_z.reshape(1, _z.shape[0]), _cond_values]).numpy()
    else:
        # Obtaining the modified spectrum
        _generated_X = model.get_layer('decoder')([_z.reshape(1, _z.shape[0])]).numpy()


    _generated_X = _generated_X if X_scaler is None else X_scaler.inverse_transform(_generated_X)

    X_source.data['generated_X'] = _generated_X[0]

def modify_slider_callback(attr, old, new):
    '''Modifies the value of the slider by the value of the last neuron in the selected latent space (because there 
    can be more than one selected)'''
    global automatic_change

    if len(new) == 0: # If the points have been deselected, the slider is not modified (there are no indexes)
        return

    logging.info('Last selected neuron value: {}'.format(z_source.data['z'][new[-1]]))
    

    automatic_change = True
    z_slider.value = z_source.data['z'][new[-1]]

def cond_param_slider_callback(attr, old, new):
    '''Generate a new spectrum with the new temperature. It can only be called if the autoencoder is conditional. In any
    other case the sliders to modify the labels are not visible.'''

    if z_source.data['z'].size == 0: # If the latent space is empty, nothing is done
        return

    _z = z_source.data['z']

    # We obtain the values of the different conditional sliders
    _slider_values = list()
    for _label in cond_params_names:
        _slider_values.append(sliders_cond_params_dict[_label].value)

    # We apply the scaler
    _cond_values = param_scaler.transform([_slider_values])


    # Obtaining the modified spectrum
    _generated_X = model.get_layer('decoder')([_z.reshape(1, _z.shape[0]), _cond_values]).numpy()

    _generated_X = _generated_X if X_scaler is None else X_scaler.inverse_transform(_generated_X)

    X_source.data['generated_X'] = _generated_X[0]

    
    # Search for an equivalent spectrum
    _df = df[(df.ids == id_selected)]
    
    _no_cond_params_labels = np.array(params_names)[~np.isin(params_names, cond_params_names)]
    for _label in _no_cond_params_labels:
        _df = _df[_df['original_' + _label] == params_selected[_label]]

    for _key, _slider in sliders_cond_params_dict.items(): # It is filtered by each parameter and its selected value
        _df = _df[_df['original_' + _key].astype(float) == _slider.value]

    if _df.shape[0] != 0: # If the combination is valid
        # We obtain the original spectrum for comparison purposes
        _original_X = np.array(_df.X.values[0])

        _original_X = _original_X if X_scaler is None else X_scaler.inverse_transform(_original_X.reshape(1,-1))

        # The comparison spectrum and the label that shows information about their APs are updated
        X_source.data['comparison_X'] = _original_X if X_scaler is None else _original_X[0]

        _text = str()
        for _label in params_names:
            _text += '{:.2f}-'.format(float(_df['original_'+_label].iloc[0]))
        comparison_label.text = _text[:-1]

        _y0 = _df.X.iloc[0][0]
        comparison_label.y = _y0 + (0.1 if _y0 < 0.5 else -0.2)
    else:
        # The label of the comparison spectrum is deleted and the spectrum itself is hidden
        comparison_label.text = ''
        X_source.data['comparison_X'] = X_source.data['comparison_X'] -10 
    

def get_spectra():
    '''Update information in figures when necessary. This is when all selector values are different from '-' '''

    if (id_selected != '-'):
        # If any of the current select parameters is '-' there is no spectrum to display
        for _param_selected in params_selected.values():
            if (_param_selected == '-'):
                return

        # Spectrum selection
        _df = df[df.ids == id_selected] # Star selection
        for _key, _param_selected in params_selected.items(): # It is filtered by each parameter and its selected value
            _df = _df[_df['original_' + _key] == _param_selected]

        if _df.shape[0] == 0: # If the combination is not valid, the figure is emptied and Teff and Logg are reset
            for _select in selects_y_dict.values():
                _select.disabled = True
                _select.value = '-'
            X_source.data = get_empty_X_source()
            z_source.data = get_empty_z_source()
        else: # Otherwise, the figures are filled in with the data
            X_source.data={
                'axis_labels': axis_labels,
                'X': np.array(_df.X.iloc[0]) if X_scaler is None else X_scaler.inverse_transform(np.array(_df.X.iloc[0]).reshape(1, -1))[0],
                'decoded_X': np.array(_df.decoded.iloc[0]) if X_scaler is None else X_scaler.inverse_transform(np.array(_df.decoded.iloc[0]).reshape(1, -1))[0],
                'generated_X': np.array(_df.decoded.iloc[0]) if X_scaler is None else X_scaler.inverse_transform(np.array(_df.decoded.iloc[0]).reshape(1, -1))[0],
                'comparison_X': np.array(_df.decoded.iloc[0]) if X_scaler is None else X_scaler.inverse_transform(np.array(_df.decoded.iloc[0]).reshape(1, -1))[0],
            }
            z_source.data={
                'z': np.array(_df.z.iloc[0]),
                'z_axis_labels': list(range(len(_df.z.iloc[0]))),
            }
            # If the model is conditional, the conditional sliders are updated 
            if is_conditional:
                for _label in cond_params_names:
                    sliders_cond_params_dict[_label].value = float(params_selected[_label])
            # Enable export button
            export_button.disabled=False
    else:
        return

def search_callback(attr, old, new):
    if (new == ''):
        ids_select.value = '-'
    else:
        new = str.upper(new)
    ids_select.options = ['-'] + [_id for _id in unique_ids if new in _id]
    return

def id_select_callback(attr, old, new):
    '''Callback that is responsible for updating the selected id. If the '-' option is selected, parameters 
    values are also reset to '-'. If a valid id is selected, the values of the parameters selectors are updated by 
    possible values for that id'''
    global id_selected

    id_selected = new # The value of the selected id is updated

    if new == old:
        return
    elif new == '-':
        for _select in selects_y_dict.values():
            _select.disabled = True
            _select.value = '-'
        
        # Empty figures
        X_source.data = get_empty_X_source()
        z_source.data = get_empty_z_source()

        # Disable export button
        export_button.disabled=True

        return

    _df = df[df.ids == new]
    
    for _label, _select in selects_y_dict.items():
        _select.disabled = False
        _unique_values = np.unique(_df['original_' + _label]).tolist()
        _select.options = ['-'] + _unique_values
        _select.value = '-'

    get_spectra()

def param_select_callback(select_obj, attr, old, new):
    '''Callback that is responsible for updating the header selects'''
    global params_selected 

    _current_param_name = select_obj.title.lower()
    params_selected[_current_param_name] = new


    if old == new:
        return
    elif new == '-':
        _df = df[(df.ids == id_selected)]
        # For each original parameter other than the selected one, the values of its select are updated
        for _label in params_names:    
            if _label == _current_param_name:
                continue
            _unique_values = np.unique(_df['original_' + _label]).tolist()
            selects_y_dict[_label].options = ['-'] + _unique_values
        
        # Empty figures
        X_source.data = get_empty_X_source()
        z_source.data = get_empty_z_source()
        
        # Disable export button
        export_button.disabled=True

        return

    _df = df[(df.ids == id_selected) & (df['original_'+_current_param_name] == new)]

    
    # For each original parameter other than the selected one, the values of its select are updated
    for _labels in params_names:    
        if _labels == _current_param_name:
            continue
        _unique_values = np.unique(_df['original_' + _labels]).tolist()
        selects_y_dict[_labels].options = ['-'] + _unique_values

    get_spectra()

def on_pan(event: Pan):
    '''Use the mouse pan function on the latent space figure to modify the y value of the selected neurons'''
    logging.info('on_pan: delta_y = ', event.y)

    _z = z_source.data['z']
    _z_axis_labels = z_source.data['z_axis_labels']

    # There is no data to modify
    if len(z_source.selected.indices) == 0: 
        return

    # Calculation of the new value
    new_value = event.y # event.y returns the value of the y-axis of the latent space where the mouse is

    # We update the slider
    z_slider.value = new_value 

    # We update the latent space ColumnDataSource
    _z[z_source.selected.indices] = new_value

    z_source.data = dict(z = _z, z_axis_labels = _z_axis_labels)

    if is_conditional:
        # We obtain the values of the different conditional sliders
        _slider_values = list()
        for _label in cond_params_names:
            _slider_values.append(sliders_cond_params_dict[_label].value)

        # We apply the scaler
        _cond_values = param_scaler.transform([_slider_values])

        # The modified spectrum is generated
        _generated_X = model.get_layer('decoder')([_z.reshape(1, _z.shape[0]), _cond_values]).numpy()
    else:
        # The modified spectrum is generated
        _generated_X = model.get_layer('decoder')([_z.reshape(1, _z.shape[0]), _cond_values]).numpy()
        

    _generated_X = _generated_X if X_scaler is None else X_scaler.inverse_transform(_generated_X)

    X_source.data['generated_X'] = _generated_X[0]

def generate_sample():
    if id_selected != '-':
        _params_str = str()
        
        # Add values of no cond params
        _no_cond_params_labels = np.array(params_names)[~np.isin(params_names, cond_params_names)]
        for _no_cond_params_label in _no_cond_params_labels:
            _params_str += '_' + _no_cond_params_label + '_' + str(selects_y_dict[_no_cond_params_label].value)

        # Add values of cond params
        for _cond_param_name in cond_params_names:
            _params_str += '_' + _cond_param_name + '_' + str(sliders_cond_params_dict[_cond_param_name].value)

        # Create folder if it does not exist
        _export_folder = os.path.join(FILE_PATH,
                                      load_configuration_data()['export_folder'])
        
        if not os.path.exists(_export_folder):
            os.makedirs(_export_folder)

        _filename = '{:}{:}'.format(id_selected, _params_str.replace('.', '_'))
        _filepath = os.path.join(_export_folder, _filename)
        # Save generated datum
        np.save(_filepath, 
                X_source.data['generated_X'])

## Linking callbacks
### Search text and header selects
search_text_input.on_change('value', search_callback)
ids_select.on_change('value', id_select_callback)

# Wrapper to be able to pass extra parameters (extra_param) to the original callback (original_callback)
def bind_select_obj(extra_param, original_callback):
    def wrapped(attr, old, new):
        original_callback(extra_param, attr, old, new)

    return wrapped

# Callbacks for the selects of the astrophysical parameters (using the wrapper) 
for _param_name in params_names:
    _select = selects_y_dict[_param_name]
    _select.on_change('value', bind_select_obj(_select, param_select_callback))

### Latent space Sliders
z_slider.on_change('value', modify_neuron_callback)
#### Conditional slider(s)
for _param_name in cond_params_names:
    _slider = sliders_cond_params_dict[_param_name]
    _slider.on_change('value', cond_param_slider_callback)

### Reset button
reset_button.on_click(get_spectra)

### Latent space representation callbacks
z_source.selected.on_change('indices', modify_slider_callback)
f2.on_event('pan', on_pan)

### Export button
export_button.on_click(generate_sample)

#######################

# Layout definition #
layout_ = column(row([search_text_input, ids_select] + list(selects_y_dict.values()), sizing_mode='scale_width'),
                 row(f1, sizing_mode='stretch_both'),
                 row(column([z_slider] + list(sliders_cond_params_dict.values()) + [reset_button], sizing_mode='stretch_height'), f2, sizing_mode='stretch_both'), 
                 f3, export_button, sizing_mode='stretch_both')

curdoc().add_root(layout_)