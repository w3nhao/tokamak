#!/usr/bin/env python

import os
import sys
import numpy as np

np.random.seed(0)

from scipy import interpolate
from common.model_structure import *
from common.wall import *
from common.setting import *
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

# Setting
wide = True
base_path = os.path.abspath(os.path.dirname(sys.argv[0]))
kstar_img_path = os.path.join(base_path, 'images', 'insideKSTAR.jpg')
max_models = 10
init_models = 1
max_shape_models = 4
seq_len = 10
decimals = np.log10(1000)
dpi = 1
plot_length = 40
t_delay = 0.05
steady_model = False
lookback = 3
show_inputs = False

n_model_box = 1

rcParamsSetting(dpi)

# Fixed setting
year_in = 2021
ec_freq = 105.e9

# Path of weights
lstm_model_path = os.path.join(base_path, 'weights', 'lstm', 'v220505')
nn_model_path = os.path.join(base_path, 'weights', 'nn')
bpw_model_path = os.path.join(base_path, 'weights', 'bpw')
k2rz_model_path = os.path.join(base_path, 'weights', 'k2rz')
rl_model_path = os.path.join(
    base_path, 
    'weights', 
    'rl', 
    'rt_control', 
    '3frame_v220505', 
    'best_model.zip', 
)


# RL setting
low_target = [0.8, 4.0, 0.80]
high_target = [2.1, 7.0, 1.05]
low_action = [0.3, 0.0, 0.0, 0.0, 1.6, 0.15, 0.5, 1.265, 2.14]
high_action = [0.8, 1.75, 1.75, 1.5, 1.95, 0.5, 0.85, 1.36, 2.3]


low_state = (low_action + low_target) * lookback + low_target
high_state = (high_action + high_target) * lookback + high_target

# Inputs
input_params = [
    'Ip [MA]', 'Bt [T]', 'GW.frac. [-]',
    'Pnb1a [MW]', 'Pnb1b [MW]', 'Pnb1c [MW]',
    'Pec2 [MW]', 'Pec3 [MW]', 'Zec2 [cm]', 'Zec3 [cm]',
    'In.Mid. [m]', 'Out.Mid. [m]', 'Elon. [-]', 'Up.Tri. [-]', 'Lo.Tri. [-]'
]
input_mins = [0.3, 1.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -10, -10, 1.265, 2.18, 1.6, 0.1, 0.5]
input_maxs = [0.8, 2.7, 0.6, 1.75, 1.75, 1.5, 0.8, 0.8, 10, 10, 1.36, 2.29, 2.0, 0.5, 0.9]
input_init = [0.5, 1.8, 0.33, 1.5, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.32, 2.22, 1.7, 0.3, 0.75]

# Outputs
output_params0 = ['βn', 'q95', 'q0', 'li']
output_params1 = ['βp', 'wmhd']
output_params2 = ['βn', 'βp', 'h89', 'h98', 'q95', 'q0', 'li', 'wmhd']
dummy_params = [
    'Ip [MA]', 'Elon. [-]', 'Up.Tri. [-]', 'Lo.Tri. [-]', 
    'In.Mid. [m]', 'Out.Mid. [m]', 'Pnb1a [MW]', 'Pnb1b [MW]', 'Pnb1c [MW]'
]

# Targets
target_params = ['βp', 'q95', 'li']
target_mins = [0.8, 4.0, 0.80]
target_maxs = [2.1, 7.0, 1.05]
target_init = [1.45, 5.5, 0.925]


# =======================
# Utility Functions
# =======================

def i2f(i, decimals=decimals):
    """Convert integer to float with fixed decimal precision."""
    return float(i/ (10 ** decimals))

def f2i(f, decimals=decimals):
    """Convert float to integer by scaling."""
    return int(f * (10 ** decimals))

# =======================
# Data Generator Class
# =======================

class KSTARDataGenerator:
    def __init__(self):
        # Initial condition
        self.first = True
        self.time = np.linspace(-0.1 * (plot_length - 1), 0, plot_length)
        self.outputs = {p: [0.0] for p in output_params2}
        self.dummy = {p: [0.0] for p in dummy_params}
        self.x = np.zeros([seq_len, 18])
        self.targets = {p: [v] * 2 for p, v in zip(target_params, target_init)}
        self.new_action = np.array(low_action)
        self.histories = [list(low_action) + list(target_init)] * lookback
        self.img = plt.imread(kstar_img_path)

        # Load models
        if steady_model:
            self.kstar_nn = kstar_nn(model_path=nn_model_path, n_models=max_models)
        else:
            self.kstar_nn = kstar_nn(model_path=nn_model_path, n_models=1)
            self.kstar_lstm = kstar_v220505(model_path=lstm_model_path, n_models=max_models)
        self.k2rz = k2rz(model_path=k2rz_model_path, n_models=max_shape_models)
        self.bpw_nn = bpw_nn(model_path=bpw_model_path, n_models=max_models)

        self.reset_model_number()
        
        # Load RL agent
        self.rl_model = SB2_model(
            model_path=rl_model_path, 
            low_state=low_state, 
            high_state=high_state, 
            low_action=low_action, 
            high_action=high_action, 
            activation='relu', 
            last_actv='tanh', 
            norm=True, 
            bavg=0.0
        )

        # Initialize inputs
        self.initialize_inputs()

    def initialize_inputs(self):
        """Initialize input parameters scaled as integers."""
        self.inputs = {param: f2i(val) for param, val in zip(input_params, input_init)}
        
    def reset_model_number(self):
        """ MImic the original codes behavior """
        if steady_model:
            self.kstar_nn.nmodels = n_model_box
        else:
            self.kstar_lstm.nmodels = n_model_box
        self.bpw_nn.nmodels = n_model_box

    def predict_boundary(self):
        """Predict plasma boundary using k2rz model."""
        ip = i2f(self.inputs['Ip [MA]'])
        bt = i2f(self.inputs['Bt [T]'])
        bp = self.outputs['βp'][-1]
        rin = i2f(self.inputs['In.Mid. [m]'])
        rout = i2f(self.inputs['Out.Mid. [m]'])
        k = i2f(self.inputs['Elon. [-]'])
        du = i2f(self.inputs['Up.Tri. [-]'])
        dl = i2f(self.inputs['Lo.Tri. [-]'])

        self.k2rz.set_inputs(ip, bt, bp, rin, rout, k, du, dl)
        self.rbdry, self.zbdry = self.k2rz.predict(post=True)
        self.rx1, self.zx1 = self.rbdry[np.argmin(self.zbdry)], np.min(self.zbdry)
        self.rx2, self.zx2 = self.rx1, -self.zx1

    def predict_0d(self, steady=True):
        """Predict 0D plasma parameters."""
        if steady:
            x = np.zeros(17)
            idx_convert = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 10, 2]
            for i in range(len(x) - 1):
                param = input_params[idx_convert[i]]
                x[i] = i2f(self.inputs[param])
            # Handle special cases
            x[9], x[10] = 0.5 * (x[9] + x[10]), 0.5 * (x[10] - x[9])
            x[14] = 1.0 if x[14] > 1.265 + 1.e-4 else 0.0
            x[-1] = year_in
            y = self.kstar_nn.predict(x)
            for i in range(len(output_params0)):
                if len(self.outputs[output_params0[i]]) >= plot_length:
                    del self.outputs[output_params0[i]][0]
                elif len(self.outputs[output_params0[i]]) == 1:
                    self.outputs[output_params0[i]][0] = y[i]
                self.outputs[output_params0[i]].append(y[i])
            self.x[:, :len(output_params0)] = y
            idx_convert = [0, 1, 2, 12, 13, 14, 10, 11, 3, 4, 5, 6, 10]
            for i in range(len(self.x[0]) - 1 - 4):
                param = input_params[idx_convert[i]]
                self.x[:, i + 4] = i2f(self.inputs[param])
            self.x[:, 11 + 4] += i2f(self.inputs[input_params[7]])
            self.x[:, 12 + 4] = 1.0 if self.x[-1, 12 + 4] > 1.265 + 1.e-4 else 0.0
            self.x[:, -1] = year_in
        else:
            self.x[:-1, len(output_params0):] = self.x[1:, len(output_params0):]
            idx_convert = [0, 1, 2, 12, 13, 14, 10, 11, 3, 4, 5, 6, 10]
            for i in range(len(self.x[0]) - 1 - 4):
                param = input_params[idx_convert[i]]
                self.x[-1, i + 4] = i2f(self.inputs[param])
            self.x[-1, 11 + 4] += i2f(self.inputs[input_params[7]])
            self.x[-1, 12 + 4] = 1.0 if self.x[-1, 12 + 4] > 1.265 + 1.e-4 else 0.0 
            y = self.kstar_lstm.predict(self.x)
            self.x[:-1, :len(output_params0)] = self.x[1:, :len(output_params0)]
            self.x[-1, :len(output_params0)] = y
            for i in range(len(output_params0)):
                if len(self.outputs[output_params0[i]]) >= plot_length:
                    del self.outputs[output_params0[i]][0]
                elif len(self.outputs[output_params0[i]]) == 1:
                    self.outputs[output_params0[i]][0] = y[i]
                self.outputs[output_params0[i]].append(y[i])
                
        # Update output targets (βp, q95, li)                
        if not self.first:
            for i,target_param in enumerate(target_params):
                if len(self.targets[target_param]) >= plot_length:
                    del self.targets[target_param][0]
                elif len(self.targets[target_param]) == 1:
                    self.targets[target_param][0] = self.targets[target_param][-1]
                self.targets[target_param].append(self.targets[target_param][-1])
                
        # Predict output_params1 (βp, wmhd)
        x = np.zeros(8)
        idx_convert = [0, 0, 1, 10, 11, 12, 13, 14]
        x[0] = self.outputs['βn'][-1]
        for i in range(1, len(x)):
            param = input_params[idx_convert[i]]
            x[i] = i2f(self.inputs[param])
        # Handle special cases
        x[3], x[4] = 0.5 * (x[3] + x[4]), 0.5 * (x[4] - x[3])
        y = self.bpw_nn.predict(x)
        for i in range(len(output_params1)):
            if len(self.outputs[output_params1[i]]) >= plot_length:
                del self.outputs[output_params1[i]][0]
            elif len(self.outputs[output_params1[i]]) == 1:
                self.outputs[output_params1[i]][0] = y[i]
            self.outputs[output_params1[i]].append(y[i])
        
        # Store dummy parameters
        for p in dummy_params:
            if len(self.dummy[p]) >= plot_length:
                del self.dummy[p][0]
            elif len(self.dummy[p]) == 1:
                self.dummy[p][0] = self.inputs[p]
            self.dummy[p].append(self.inputs[p])

        # Update histories
        self.histories[:-1] = self.histories[1:]
        self.histories[-1] = list(self.new_action) + list([
            self.outputs['βp'][-1], 
            self.outputs['q95'][-1], 
            self.outputs['li'][-1]
        ])

        # Estimate H factors (h89, h98)
        ip = i2f(self.inputs['Ip [MA]'])
        bt = i2f(self.inputs['Bt [T]'])
        fgw = i2f(self.inputs['GW.frac. [-]'])
        ptot = max(
            i2f(self.inputs['Pnb1a [MW]']) +
            i2f(self.inputs['Pnb1b [MW]']) +
            i2f(self.inputs['Pnb1c [MW]']) +
            i2f(self.inputs['Pec2 [MW]']) +
            i2f(self.inputs['Pec3 [MW]']),
            1.e-1
        )  # Prevent division by zero
        rin = i2f(self.inputs['In.Mid. [m]'])
        rout = i2f(self.inputs['Out.Mid. [m]'])
        k = i2f(self.inputs['Elon. [-]'])
        rgeo, amin = 0.5 * (rin + rout), 0.5 * (rout - rin)
        ne = fgw * 10 * (ip / (np.pi * amin**2))
        m = 2.0  # Mass number

        tau89 = 0.038 * ip**0.85 * bt**0.2 * ne**0.1 * ptot**-0.5 * rgeo**1.5 * k**0.5 * (amin / rgeo)**0.3 * m**0.5
        tau98 = 0.0562 * ip**0.93 * bt**0.15 * ne**0.41 * ptot**-0.69 * rgeo**1.97 * k**0.78 * (amin / rgeo)**0.58 * m**0.19
        h89 = 1.e-6 * self.outputs['wmhd'][-1] / ptot / tau89
        h98 = 1.e-6 * self.outputs['wmhd'][-1] / ptot / tau98
        
        if len(self.outputs['h89']) >= plot_length:
            del self.outputs['h89'][0], self.outputs['h98'][0]
        elif len(self.outputs['h89']) == 1:
            self.outputs['h89'][0], self.outputs['h98'][0] = h89, h98

        self.outputs['h89'].append(h89)
        self.outputs['h98'].append(h98)


    def auto_control(self):
        """Use RL model to predict new actions and update inputs."""
        observation = np.zeros_like(low_state)
        for i in range(lookback):
            observation[i * len(self.histories[0]): (i + 1) * len(self.histories[0])] = self.histories[i]
            
        # Targets are the last entries
        observation[lookback * len(self.histories[0]):] = [
            self.targets[target_params[j]][-1] for j in range(len(target_params))
        ]
        self.new_action = self.rl_model.predict(observation, yold=self.new_action)
        
        idx_convert = [0, 3, 4, 5, 12, 13, 14, 10, 11]
        for i, idx in enumerate(idx_convert):
            param = input_params[idx]
            # Ensure the new action is within the action bounds
            action_value = np.clip(self.new_action[i], low_action[i], high_action[i])
            self.inputs[param] = f2i(action_value)


    def data_collect(self):
        """Collect data for the simulation."""
        record = {}
        # Input Parameters (Convert back to float)
        for param in input_params:
            record["in_" + param] = i2f(self.inputs[param])
        # Output Parameters
        for param in output_params2:
            record["out2_" + param] = self.outputs[param][-1]
        # Dummy Parameters
        for param in dummy_params:
            record["dum_" + param] = self.dummy[param][-1]
        for param in target_params:
            record["tgt_" + param] = self.targets[param][-1]

        return record


    def run_simulation(self, seconds=10):
        """Run the simulation for a specified number of steps."""
        data_records = []
        
        # Predict boundary and 0D parameters first step 
        self.predict_boundary()
        self.predict_0d(steady=True)
        self.first = False
        
        record = self.data_collect()
        data_records.append(record)
        
        # update the target values
        self.auto_control()
        self.predict_boundary()
        self.predict_0d(steady=steady_model)
        
        record = self.data_collect()
        data_records.append(record)
        
        p_bar = tqdm(total=seconds * 10)
        
        # steps = int(seconds / t_delay)
        for sec in range(seconds):
            for step in range(10 - 1):
                # Auto control every 'control_interval' steps
                self.auto_control()

                # Predict 0D parameters
                self.predict_0d(steady=steady_model)
                
                p_bar.update(1)
            
            # Update the target values
            self.auto_control()
            self.predict_boundary()
            self.predict_0d(steady=steady_model)
            
            p_bar.update(1)

            # Collect data: Convert inputs back to floats for recording
            record = self.data_collect()
            data_records.append(record)

            # Optional: Add delay if needed
            # time.sleep(t_delay)

        # Save data to numpy
        data_records = np.array(data_records)
        output_file = os.path.join(base_path, 'kstar_generated_data.npz')
        # compress data
        np.savez_compressed(output_file, data=data_records)
        print(f'Data generation complete. Saved to {output_file}')

# =======================
# Main Execution
# =======================

if __name__ == '__main__':
    generator = KSTARDataGenerator()
    generator.run_simulation(seconds=10)
