import os
import sys
import numpy as np
import logging
from bokeh.models import ColumnDataSource, TapTool, Slider, Select, Button, TextInput, Label
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.events import Pan

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.normpath('../../'))) # To support the use of the tool without packaging
from gandalf.visualization.load_utils import load_configuration_data, load_data, load_model, load_scalers

## GCD ##
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

class GandalfVisualiser():

    def __init__(self, file_path: str):
        self.file_path = file_path

    def set_up_data(self):
        # attributes to be used in the callbacks #
        self.automatic_change = False
        self.id_selected = '-'
        self.params_selected = dict()

        # attributes from results/data folder#
        self.model = load_model(self.file_path)
        self.is_conditional = True

        self.df, self.axis_labels, self.params_names, self.cond_params_names = load_data(self.file_path)

        self.X_scaler, self.param_scaler = load_scalers(self.file_path)

        self.export_folder = os.path.join(load_configuration_data(self.file_path)['export_folder'])

        # A dictionary is created for the original parameters with their unique values
        self.unique_original_dict = dict()
        for _param_name in self.params_names:
            _rounded_param = [round(float(_param), 2) for _param in self.df['original_' + _param_name].values]
            self.unique_original_dict[_param_name] = np.unique(_rounded_param).tolist()

        self.unique_ids = np.unique(self.df.ids).tolist()

        # A dictionary is created for the current values of each select of the astrophysical parameters
        for _param_name in self.params_names:
            self.params_selected[_param_name] = '-'

        # Creation of the ColumnDataSource (as all fields must have the same length, a ColumnDataSource is needed for the latent space and another for the spectra)
        self.X_source = ColumnDataSource(data=get_empty_X_source())

        self.z_source = ColumnDataSource(data=get_empty_z_source())

    def set_up_widgets(self):
        ### TextInput
        self.search_text_input = TextInput(title='Search by id:', value='', placeholder='identifier')
        ### Selects
        self.ids_select = Select(title='Identifier', value=None, options=['-'] + self.unique_ids)
        #### Selects for parameters, we create a dictionary that contains the different selects of astrophysical parameters
        self.selects_y_dict = dict() # Select dict
        for _param_name in self.params_names:
            _select = Select(title=_param_name, value='', disabled=True)
            self.selects_y_dict[_param_name] = _select

        ### Reset button
        self.reset_button = Button(label='Reset', button_type='danger',
                            width=150, height=30, sizing_mode='fixed')

        ### Sliders (https://docs.bokeh.org/en/latest/docs/user_guide/interaction/widgets.html#slider)
        self.z_slider = Slider(start=np.min(np.min(self.df.z.values)), end=np.max(np.max(self.df.z.values)), value=0, step=0.05, title='Selected Neuron Value',
                        width=150, height=50, sizing_mode='fixed')
        self.z_slider.disabled = True

        #### Sliders for conditional parameters
        self.sliders_cond_params_dict = dict() # Slider dict

        for _cond_param_name in self.cond_params_names:

            _unique_values = [float(_value) for _value in self.unique_original_dict[_cond_param_name]] # se convierte str a float para inicializar los valores del slider

            _slider = Slider(title=_cond_param_name, start=_unique_values[0], end=_unique_values[-1], value=np.median(_unique_values),
                            step=gcd_list(_unique_values),
                            width=150, height=50, sizing_mode='fixed')

            self.sliders_cond_params_dict[_cond_param_name] = _slider

        ### Export button
        self.export_button = Button(label='Generate sample', button_type='success', align='end',
                            width=150, height=30, sizing_mode='fixed', margin=(5,40,5,5), disabled=True)

    def set_up_figures(self):
        ## Plot original spectrum
        _X = np.array(self.df.X.values[0])
        y_range_min = np.min(_X) if self.X_scaler is None else np.min(self.X_scaler.inverse_transform(_X.reshape(1, -1)))
        y_range_max = np.max(_X) if self.X_scaler is None else np.max(self.X_scaler.inverse_transform(_X.reshape(1, -1)))
        self.f1 = figure(width=18, height=3, sizing_mode='scale_width', tools=['pan', 'box_zoom', 'xwheel_zoom', 'reset'],
                    y_range=(y_range_min, y_range_max), output_backend='webgl')
        self.f1.line(x='axis_labels', y='X', source=self.X_source, line_color='#ECA400', line_width=2)
        self.f1.toolbar.autohide=True


        ## Plot latent
        self.f2 = figure(width=18, height=3, sizing_mode='scale_width', tools='', output_backend='webgl')
        self.f2.line(x='z_axis_labels', y ='z', source=self.z_source)
        self.f2.scatter(x='z_axis_labels', y='z', source=self.z_source, size=8, marker='circle', line_color='navy', fill_color='orange', fill_alpha=1)

        self.f2.add_tools(TapTool())
        self.f2.toolbar.autohide=True


        ## Plot decoded spectrum
        self.f3 = figure(width=18, height=3, x_range=self.f1.x_range, y_range=self.f1.y_range,
                    sizing_mode='scale_width', output_backend='webgl')
        self.f3.line(x='axis_labels', y='comparison_X', source=self.X_source, line_color='#D5573B', line_width=2,
                legend_label='original')
        self.f3.line(x='axis_labels', y='generated_X', source=self.X_source, line_color='#61988E', line_width=2,
                legend_label='generated')
        self.f3.line(x='axis_labels', y='decoded_X', source=self.X_source, line_color='#27476E', line_width=2,
                legend_label='decoded')

        self.comparison_label = Label(x=self.axis_labels[0], y=-1, x_units='data', y_units='data',
                                text='', text_color='red',
                                border_line_color='white', border_line_width=5,
                                background_fill_color='white',
                                text_font_size='18px')
        self.f3.add_layout(self.comparison_label)
        self.f3.toolbar.autohide=True

        ## Common figure properties
        for f in [self.f1, self.f2, self.f3]:
            f.toolbar.autohide=True
            f.background_fill_color = (250, 250, 250)
        ####################

    # CALLBACKS #

    def modify_neuron_callback(self, attr, old, new):
        '''Replaces the value of the selected neurons in the latent space with the value selected in the slider'''
        logging.info('Slider value: {}'.format(new))
        logging.info('Indices: {}'.format(self.z_source.selected.indices))

        if self.X_source.data['X'].size == 0: # If the arrays are empty, nothing is done (just check one)
            return

        if self.automatic_change:
            self.automatic_change = False
            return

        _z = self.z_source.data['z']
        _z_axis_labels = self.z_source.data['z_axis_labels']

        _z[self.z_source.selected.indices] = new

        self.z_source.data = dict(z = _z, z_axis_labels = _z_axis_labels)

        if self.is_conditional:
            # We obtain the values of the different conditional sliders
            _slider_values = list()
            for _label in self.cond_params_names:
                _slider_values.append(self.sliders_cond_params_dict[_label].value)

            # We apply the scaler
            _cond_values = self.param_scaler.transform([_slider_values])

            # Obtaining the modified spectrum
            _generated_X = self.model.get_layer('decoder')([_z.reshape(1, _z.shape[0]), _cond_values]).numpy()
        else:
            # Obtaining the modified spectrum
            _generated_X = self.model.get_layer('decoder')([_z.reshape(1, _z.shape[0])]).numpy()


        _generated_X = _generated_X if self.X_scaler is None else self.X_scaler.inverse_transform(_generated_X)

        self.X_source.data['generated_X'] = _generated_X[0]

    def modify_slider_callback(self, attr, old, new):
        '''Modifies the value of the slider by the value of the last neuron in the selected latent space (because there 
        can be more than one selected)'''

        if len(new) == 0: # If the points have been deselected, the slider is not modified (there are no indexes)
            self.z_slider.disabled = True
            return
        self.z_slider.disabled = False
        logging.info('Last selected neuron value: {}'.format(self.z_source.data['z'][new[-1]]))

        self.automatic_change = True
        self.z_slider.value = self.z_source.data['z'][new[-1]]

    def cond_param_slider_callback(self, attr, old, new):
        '''Generate a new spectrum with the new temperature. It can only be called if the autoencoder is conditional. In any
        other case the sliders to modify the labels are not visible.'''

        if self.z_source.data['z'].size == 0: # If the latent space is empty, nothing is done
            return

        _z = self.z_source.data['z']

        # We obtain the values of the different conditional sliders
        _slider_values = list()
        for _label in self.cond_params_names:
            _slider_values.append(self.sliders_cond_params_dict[_label].value)

        # We apply the scaler
        _cond_values = self.param_scaler.transform([_slider_values])


        # Obtaining the modified spectrum
        _generated_X = self.model.get_layer('decoder')([_z.reshape(1, _z.shape[0]), _cond_values]).numpy()

        _generated_X = _generated_X if self.X_scaler is None else self.X_scaler.inverse_transform(_generated_X)

        self.X_source.data['generated_X'] = _generated_X[0]


        # Search for an equivalent spectrum
        _df = self.df[(self.df.ids == self.id_selected)]

        _no_cond_params_labels = np.array(self.params_names)[~np.isin(self.params_names, self.cond_params_names)]
        for _label in _no_cond_params_labels:
            _df = _df[_df['original_' + _label] == self.params_selected[_label]]

        for _key, _slider in self.sliders_cond_params_dict.items(): # It is filtered by each parameter and its selected value
            _df = _df[_df['original_' + _key].astype(float) == _slider.value]

        if _df.shape[0] != 0: # If the combination is valid
            # We obtain the original spectrum for comparison purposes
            _original_X = np.array(_df.X.values[0])

            _original_X = _original_X if self.X_scaler is None else self.X_scaler.inverse_transform(_original_X.reshape(1,-1))

            # The comparison spectrum and the label that shows information about their APs are updated
            self.X_source.data['comparison_X'] = _original_X if self.X_scaler is None else _original_X[0]

            _text = str()
            for _label in self.params_names:
                _text += '{:.2f}-'.format(float(_df['original_'+_label].iloc[0]))
            self.comparison_label.text = _text[:-1]

            _y0 = _df.X.iloc[0][0]
            self.comparison_label.y = _y0 + (0.1 if _y0 < 0.5 else -0.2)
        else:
            # The label of the comparison spectrum is deleted and the spectrum itself is hidden
            self.comparison_label.text = ''
            self.X_source.data['comparison_X'] = self.X_source.data['comparison_X'] -10

    def get_spectra(self):
        '''Update information in figures when necessary. This is when all selector values are different from '-' '''
        self.z_slider.disabled = True
        if (self.id_selected != '-'):
            # If any of the current select parameters is '-' there is no spectrum to display
            for _param_selected in self.params_selected.values():
                if (_param_selected == '-'):
                    return

            # Spectrum selection
            _df = self.df[self.df.ids == self.id_selected] # Star selection
            for _key, _param_selected in self.params_selected.items(): # It is filtered by each parameter and its selected value
                _df = _df[_df['original_' + _key] == _param_selected]

            if _df.shape[0] == 0: # If the combination is not valid, the figure is emptied and Teff and Logg are reset
                for _select in self.selects_y_dict.values():
                    _select.disabled = True
                    _select.value = '-'
                self.X_source.data = get_empty_X_source()
                self.z_source.data = get_empty_z_source()
            else: # Otherwise, the figures are filled in with the data
                self.X_source.data={
                    'axis_labels': self.axis_labels,
                    'X': np.array(_df.X.iloc[0]) if self.X_scaler is None else self.X_scaler.inverse_transform(np.array(_df.X.iloc[0]).reshape(1, -1))[0],
                    'decoded_X': np.array(_df.decoded.iloc[0]) if self.X_scaler is None else self.X_scaler.inverse_transform(np.array(_df.decoded.iloc[0]).reshape(1, -1))[0],
                    'generated_X': np.array(_df.decoded.iloc[0]) if self.X_scaler is None else self.X_scaler.inverse_transform(np.array(_df.decoded.iloc[0]).reshape(1, -1))[0],
                    'comparison_X': np.array(_df.decoded.iloc[0]) if self.X_scaler is None else self.X_scaler.inverse_transform(np.array(_df.decoded.iloc[0]).reshape(1, -1))[0],
                }
                self.z_source.data={
                    'z': np.array(_df.z.iloc[0]),
                    'z_axis_labels': list(range(len(_df.z.iloc[0]))),
                }
                # If the model is conditional, the conditional sliders are updated
                if self.is_conditional:
                    for _label in self.cond_params_names:
                        self.sliders_cond_params_dict[_label].value = float(self.params_selected[_label])
                # Enable export button
                self.export_button.disabled=False
        else:
            return

    def search_callback(self, attr, old, new):
        if (new == ''):
            self.ids_select.value = '-'
        else:
            new = str.upper(new)
        self.ids_select.options = ['-'] + [_id for _id in self.unique_ids if new in _id]
        return

    def id_select_callback(self, attr, old, new):
        '''Callback that is responsible for updating the selected id. If the '-' option is selected, parameters 
        values are also reset to '-'. If a valid id is selected, the values of the parameters selectors are updated by 
        possible values for that id'''

        self.id_selected = new # The value of the selected id is updated

        if new == old:
            return
        elif new == '-':
            for _select in self.selects_y_dict.values():
                _select.disabled = True
                _select.value = '-'

            # Empty figures
            self.X_source.data = get_empty_X_source()
            self.z_source.data = get_empty_z_source()

            # Disable export button
            self.export_button.disabled=True

            return

        _df = self.df[self.df.ids == new]

        for _label, _select in self.selects_y_dict.items():
            _unique_values = np.unique(_df['original_' + _label]).tolist()
            if len(_unique_values) > 1:
                _select.disabled = False
                _select.options = ['-'] + _unique_values
                _select.value = '-'
            elif len(_unique_values) == 1:
                _select.disabled = True
                _select.options = _unique_values
                _select.value = _unique_values[0]
            else: # This case cannot happen, it would be a bug to fix
                print('ERROR: No unique values for parameter: {}'.format(_label))
                _select.disabled = True
                _select.options = ['-']
                _select.value = '-'

        self.get_spectra()

    def param_select_callback(self, select_obj, attr, old, new):
        '''Callback that is responsible for updating the header selects'''

        _current_param_name = select_obj.title.lower()
        self.params_selected[_current_param_name] = new

        if old == new:
            return
        elif new == '-':
            _df = self.df[(self.df.ids == self.id_selected)]
            # For each original parameter other than the selected one, the values of its select are updated
            for _label in self.params_names:
                if _label == _current_param_name:
                    continue
                _unique_values = np.unique(_df['original_' + _label]).tolist()
                self.selects_y_dict[_label].options = ['-'] + _unique_values

            # Empty figures
            self.X_source.data = get_empty_X_source()
            self.z_source.data = get_empty_z_source()

            # Disable export button
            self.export_button.disabled=True

            return

        _df = self.df[(self.df.ids == self.id_selected)]

        # Filter the dataframe based on the values of the parameters already selected
        for _key, _param_selected in self.params_selected.items():
            if (_param_selected == '-'):
                continue
            _df = _df[_df['original_' + _key] == _param_selected]


        # For each original parameter other than the selected one, the values of its select are updated
        for _labels in self.params_names:
            if _labels == _current_param_name:
                continue
            _unique_values = np.unique(_df['original_' + _labels]).tolist()
            self.selects_y_dict[_labels].options = ['-'] + _unique_values

        self.get_spectra()

    def on_pan(self, event: Pan):
        '''Use the mouse pan function on the latent space figure to modify the y value of the selected neurons'''
        logging.info('on_pan: delta_y = ', event.y)

        _z = self.z_source.data['z']
        _z_axis_labels = self.z_source.data['z_axis_labels']

        # There is no data to modify
        if len(self.z_source.selected.indices) == 0:
            return

        # Calculation of the new value
        new_value = event.y # event.y returns the value of the y-axis of the latent space where the mouse is

        # We update the slider
        self.z_slider.disabled = False
        self.z_slider.value = new_value

        # We update the latent space ColumnDataSource
        _z[self.z_source.selected.indices] = new_value

        self.z_source.data = dict(z = _z, z_axis_labels = _z_axis_labels)

        if self.is_conditional:
            # We obtain the values of the different conditional sliders
            _slider_values = list()
            for _label in self.cond_params_names:
                _slider_values.append(self.sliders_cond_params_dict[_label].value)

            # We apply the scaler
            _cond_values = self.param_scaler.transform([_slider_values])

            # The modified spectrum is generated
            _generated_X = self.model.get_layer('decoder')([_z.reshape(1, _z.shape[0]), _cond_values]).numpy()
        else:
            # The modified spectrum is generated
            _generated_X = self.model.get_layer('decoder')([_z.reshape(1, _z.shape[0]), _cond_values]).numpy()


        _generated_X = _generated_X if self.X_scaler is None else self.X_scaler.inverse_transform(_generated_X)

        self.X_source.data['generated_X'] = _generated_X[0]

    def generate_sample(self):
        if self.id_selected != '-':
            _params_str = str()

            # Add values of no cond params
            _no_cond_params_labels = np.array(self.params_names)[~np.isin(self.params_names, self.cond_params_names)]
            for _no_cond_params_label in _no_cond_params_labels:
                _params_str += '_' + _no_cond_params_label + '_' + str(self.selects_y_dict[_no_cond_params_label].value)

            # Add values of cond params
            for _cond_param_name in self.cond_params_names:
                _params_str += '_' + _cond_param_name + '_' + str(self.sliders_cond_params_dict[_cond_param_name].value)

            if not os.path.exists(self.export_folder):
                os.makedirs(self.export_folder)

            _filename = '{:}{:}'.format(self.id_selected, _params_str.replace('.', '_'))
            _filepath = os.path.join(self.export_folder, _filename)
            # Save generated datum
            np.save(_filepath,
                    self.X_source.data['generated_X'])

    def link_callbacks(self):

        ### Search text and header selects
        self.search_text_input.on_change('value', self.search_callback)
        self.ids_select.on_change('value', self.id_select_callback)

        # Wrapper to be able to pass extra parameters (extra_param) to the original callback (original_callback)
        def bind_select_obj(extra_param, original_callback):
            def wrapped(attr, old, new):
                original_callback(extra_param, attr, old, new)

            return wrapped

        # Callbacks for the selects of the astrophysical parameters (using the wrapper)
        for _param_name in self.params_names:
            _select = self.selects_y_dict[_param_name]
            _select.on_change('value', bind_select_obj(_select, self.param_select_callback))

        ### Latent space Sliders
        self.z_slider.on_change('value', self.modify_neuron_callback)
        #### Conditional slider(s)
        for _param_name in self.cond_params_names:
            _slider = self.sliders_cond_params_dict[_param_name]
            _slider.on_change('value', self.cond_param_slider_callback)

        ### Reset button
        self.reset_button.on_click(self.get_spectra)

        ### Latent space representation callbacks
        self.z_source.selected.on_change('indices', self.modify_slider_callback)
        self.f2.on_event('pan', self.on_pan)

        ### Export button
        self.export_button.on_click(self.generate_sample)

    def get_layout(self):

        self.set_up_data()

        self.set_up_widgets()

        self.set_up_figures()

        self.link_callbacks()

        search = row([self.search_text_input, self.ids_select] + list(self.selects_y_dict.values()), sizing_mode='scale_width')
        buttons_and_z = row(column([self.z_slider] + list(self.sliders_cond_params_dict.values()) + [self.reset_button], sizing_mode='stretch_height'), self.f2, sizing_mode='scale_width')
        layout_ = column(search, self.f1, buttons_and_z, self.f3, self.export_button, sizing_mode='stretch_both')

        return layout_
