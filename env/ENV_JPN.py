import os
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MSWEnv(gym.Env):
    def __init__(self, data_file, reward_mode='carbon', composition_range=None, total_mass_range=None):
        
        self.disposal_params, self.transport_params = self.load_params_from_excel(data_file)
        self.reward_mode = reward_mode
        
        self.composition_keys = ['food', 'glass', 'metal', 'paper', 'plastic', 'rubber', 'wood', 'yard', 'other']
        self.composition_range = None  
        self.total_mass_range = None   
        
        
        self.total_mass = None
        self.composition = None
        
        self.composition_vector = np.zeros(9, dtype=np.float32)
        
        orig_obs = 29
        new_obs = orig_obs + len(self.composition_vector) + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(new_obs,), dtype=np.float32
        )
        
        
        self.transport_distance = self.transport_params['distance']   
        self.transport_carbon = self.transport_params['carbon_per_ton_km']
        self.transport_energy = self.transport_params['energy_per_ton_km']
        
        
        
        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(29,), dtype=np.float32)
        
        
        self.max_steps = 20  
        self.current_step = 0
        self.state = None
        self.prev_energy = float('inf')  
        self.prev_cost = float('inf')    
        self.episode_reward = 0.0  

        # ==== Biomass-led config ====
        # IP4: [incineration, landfill, AD, composting, other_biotech, D&AF, sewer]
        self.BIOMASS_IP4_IDX = [2, 3, 4, 5]  
        
        self.BIOMASS_IP11_IDX = [0]          

        
        self.food_share_for_weight = getattr(self, "food_share_for_weight", 1.0)
        self.ip11_share_for_weight = getattr(self, "ip11_share_for_weight", 1.0)

        
        self.prev_violation = None
        self.prev_ip4 = None
        self.prev_ip11 = None
        self.no_progress_streak = 0  


    def load_params_from_excel(self, file_path):
        df = pd.read_excel(file_path, sheet_name='per_JPN', header=None)
        disposal_params = {
            'open_burning':    {'energy_input': df.iat[12, 2], 'energy_input_carbon': df.iat[16, 2],
                                'op_carbon': df.iat[19, 2], 'energy_output': df.iat[23, 2],
                                'energy_output_carbon_reduction': df.iat[27, 2],
                                'resource_output': df.iat[41, 2], 'resource_output_carbon_reduction': df.iat[51, 2],
                                'cost': 10},
            'open_dumping':    {'energy_input': df.iat[12, 3], 'energy_input_carbon': df.iat[16, 3],
                                'op_carbon': df.iat[19, 3], 'energy_output': df.iat[23, 3],
                                'energy_output_carbon_reduction': df.iat[27, 3],
                                'resource_output': df.iat[41, 3], 'resource_output_carbon_reduction': df.iat[51, 3],
                                'cost': 10},
            'unsorted_incineration': {'energy_input': df.iat[12, 4], 'energy_input_carbon': df.iat[16, 4],
                                      'op_carbon': df.iat[19, 4], 'energy_output': df.iat[23, 4],
                                      'energy_output_carbon_reduction': df.iat[27, 4],
                                      'resource_output': df.iat[41, 4], 'resource_output_carbon_reduction': df.iat[51, 4],
                                      'cost': 129.36},
            'unsorted_landfill':     {'energy_input': df.iat[12, 5], 'energy_input_carbon': df.iat[16, 5],
                                      'op_carbon': df.iat[19, 5], 'energy_output': df.iat[23, 5],
                                      'energy_output_carbon_reduction': df.iat[27, 5],
                                      'resource_output': df.iat[41, 5], 'resource_output_carbon_reduction': df.iat[51, 5],
                                      'cost': 82.51},
            'food_incineration':    {'energy_input': df.iat[12, 7], 'energy_input_carbon': df.iat[16, 7],
                                     'op_carbon': df.iat[19, 7], 'energy_output': df.iat[23, 7],
                                     'energy_output_carbon_reduction': df.iat[27, 7],
                                     'resource_output': df.iat[41, 7], 'resource_output_carbon_reduction': df.iat[51, 7],
                                     'cost': 129.36},
            'food_landfill':        {'energy_input': df.iat[12, 8], 'energy_input_carbon': df.iat[16, 8],
                                     'op_carbon': df.iat[19, 8], 'energy_output': df.iat[23, 8],
                                     'energy_output_carbon_reduction': df.iat[27, 8],
                                     'resource_output': df.iat[41, 8], 'resource_output_carbon_reduction': df.iat[51, 8],
                                     'cost': 82.51},
            'food_AD':              {'energy_input': df.iat[12, 9], 'energy_input_carbon': df.iat[16, 9],
                                     'op_carbon': df.iat[19, 9], 'energy_output': df.iat[23, 9],
                                     'energy_output_carbon_reduction': df.iat[27, 9],
                                     'resource_output': df.iat[41, 9], 'resource_output_carbon_reduction': df.iat[51, 9],
                                     'cost': 91.77},
            'food_composting':      {'energy_input': df.iat[12, 10], 'energy_input_carbon': df.iat[16, 10],
                                     'op_carbon': df.iat[19, 10], 'energy_output': df.iat[23, 10],
                                     'energy_output_carbon_reduction': df.iat[27, 10],
                                     'resource_output': df.iat[41, 10], 'resource_output_carbon_reduction': df.iat[51, 10],
                                     'cost': 73.67},
            'food_other_biotech':   {'energy_input': df.iat[12, 11], 'energy_input_carbon': df.iat[16, 11],
                                     'op_carbon': df.iat[19, 11], 'energy_output': df.iat[23, 11],
                                     'energy_output_carbon_reduction': df.iat[27, 11],
                                     'resource_output': df.iat[41, 11], 'resource_output_carbon_reduction': df.iat[51, 11],
                                     'cost': 91.77},
            'food_D&AF':            {'energy_input': df.iat[12, 12], 'energy_input_carbon': df.iat[16, 12],
                                     'op_carbon': df.iat[19, 12], 'energy_output': df.iat[23, 12],
                                     'energy_output_carbon_reduction': df.iat[27, 12],
                                     'resource_output': df.iat[41, 12], 'resource_output_carbon_reduction': df.iat[51, 12],
                                     'cost': 170.91},
            'food_sewer':           {'energy_input': df.iat[12, 13], 'energy_input_carbon': df.iat[16, 13],
                                     'op_carbon': df.iat[19, 13], 'energy_output': df.iat[23, 13],
                                     'energy_output_carbon_reduction': df.iat[27, 13],
                                     'resource_output': df.iat[41, 13], 'resource_output_carbon_reduction': df.iat[51, 13],
                                     'cost': 91.77},
            'glass_recycle':        {'energy_input': df.iat[12, 15], 'energy_input_carbon': df.iat[16, 15],
                                     'op_carbon': df.iat[19, 15], 'energy_output': df.iat[23, 15],
                                     'energy_output_carbon_reduction': df.iat[27, 15],
                                     'resource_output': df.iat[41, 15], 'resource_output_carbon_reduction': df.iat[51, 15],
                                     'cost': 64.83},
            'glass_landfill':       {'energy_input': df.iat[12, 16], 'energy_input_carbon': df.iat[16, 16],
                                     'op_carbon': df.iat[19, 16], 'energy_output': df.iat[23, 16],
                                     'energy_output_carbon_reduction': df.iat[27, 16],
                                     'resource_output': df.iat[41, 16], 'resource_output_carbon_reduction': df.iat[51, 16],
                                     'cost': 82.51},
            'metal_recycle':        {'energy_input': df.iat[12, 18], 'energy_input_carbon': df.iat[16, 18],
                                     'op_carbon': df.iat[19, 18], 'energy_output': df.iat[23, 18],
                                     'energy_output_carbon_reduction': df.iat[27, 18],
                                     'resource_output': df.iat[41, 18], 'resource_output_carbon_reduction': df.iat[51, 18],
                                     'cost': 64.83},
            'metal_landfill':       {'energy_input': df.iat[12, 19], 'energy_input_carbon': df.iat[16, 19],
                                     'op_carbon': df.iat[19, 19], 'energy_output': df.iat[23, 19],
                                     'energy_output_carbon_reduction': df.iat[27, 19],
                                     'resource_output': df.iat[41, 19], 'resource_output_carbon_reduction': df.iat[51, 19],
                                     'cost': 82.51},
            'paper_recycle':        {'energy_input': df.iat[12, 21], 'energy_input_carbon': df.iat[16, 21],
                                     'op_carbon': df.iat[19, 21], 'energy_output': df.iat[23, 21],
                                     'energy_output_carbon_reduction': df.iat[27, 21],
                                     'resource_output': df.iat[41, 21], 'resource_output_carbon_reduction': df.iat[51, 21],
                                     'cost': 64.83},
            'paper_incineration':   {'energy_input': df.iat[12, 22], 'energy_input_carbon': df.iat[16, 22],
                                     'op_carbon': df.iat[19, 22], 'energy_output': df.iat[23, 22],
                                     'energy_output_carbon_reduction': df.iat[27, 22],
                                     'resource_output': df.iat[41, 22], 'resource_output_carbon_reduction': df.iat[51, 22],
                                     'cost': 129.36},
            'paper_landfill':       {'energy_input': df.iat[12, 23], 'energy_input_carbon': df.iat[16, 23],
                                     'op_carbon': df.iat[19, 23], 'energy_output': df.iat[23, 23],
                                     'energy_output_carbon_reduction': df.iat[27, 23],
                                     'resource_output': df.iat[41, 23], 'resource_output_carbon_reduction': df.iat[51, 23],
                                     'cost': 82.51},
            'plastic_recycle':      {'energy_input': df.iat[12, 25], 'energy_input_carbon': df.iat[16, 25],
                                     'op_carbon': df.iat[19, 25], 'energy_output': df.iat[23, 25],
                                     'energy_output_carbon_reduction': df.iat[27, 25],
                                     'resource_output': df.iat[41, 25], 'resource_output_carbon_reduction': df.iat[51, 25],
                                     'cost': 64.83},
            'plastic_incineration': {'energy_input': df.iat[12, 26], 'energy_input_carbon': df.iat[16, 26],
                                     'op_carbon': df.iat[19, 26], 'energy_output': df.iat[23, 26],
                                     'energy_output_carbon_reduction': df.iat[27, 26],
                                     'resource_output': df.iat[41, 26], 'resource_output_carbon_reduction': df.iat[51, 26],
                                     'cost': 129.36},
            'plastic_landfill':     {'energy_input': df.iat[12, 27], 'energy_input_carbon': df.iat[16, 27],
                                     'op_carbon': df.iat[19, 27], 'energy_output': df.iat[23, 27],
                                     'energy_output_carbon_reduction': df.iat[27, 27],
                                     'resource_output': df.iat[41, 27], 'resource_output_carbon_reduction': df.iat[51, 27],
                                     'cost': 82.51},
            'rubber_recycle':       {'energy_input': df.iat[12, 29], 'energy_input_carbon': df.iat[16, 29],
                                     'op_carbon': df.iat[19, 29], 'energy_output': df.iat[23, 29],
                                     'energy_output_carbon_reduction': df.iat[27, 29],
                                     'resource_output': df.iat[41, 29], 'resource_output_carbon_reduction': df.iat[51, 29],
                                     'cost': 64.83},
            'rubber_incineration':  {'energy_input': df.iat[12, 30], 'energy_input_carbon': df.iat[16, 30],
                                     'op_carbon': df.iat[19, 30], 'energy_output': df.iat[23, 30],
                                     'energy_output_carbon_reduction': df.iat[27, 30],
                                     'resource_output': df.iat[41, 30], 'resource_output_carbon_reduction': df.iat[51, 30],
                                     'cost': 129.36},
            'rubber_landfill':      {'energy_input': df.iat[12, 31], 'energy_input_carbon': df.iat[16, 31],
                                     'op_carbon': df.iat[19, 31], 'energy_output': df.iat[23, 31],
                                     'energy_output_carbon_reduction': df.iat[27, 31],
                                     'resource_output': df.iat[41, 31], 'resource_output_carbon_reduction': df.iat[51, 31],
                                     'cost': 82.51},
            'wood_recycle':         {'energy_input': df.iat[12, 33], 'energy_input_carbon': df.iat[16, 33],
                                     'op_carbon': df.iat[19, 33], 'energy_output': df.iat[23, 33],
                                     'energy_output_carbon_reduction': df.iat[27, 33],
                                     'resource_output': df.iat[41, 33], 'resource_output_carbon_reduction': df.iat[51, 33],
                                     'cost': 64.83},
            'wood_incineration':    {'energy_input': df.iat[12, 34], 'energy_input_carbon': df.iat[16, 34],
                                     'op_carbon': df.iat[19, 34], 'energy_output': df.iat[23, 34],
                                     'energy_output_carbon_reduction': df.iat[27, 34],
                                     'resource_output': df.iat[41, 34], 'resource_output_carbon_reduction': df.iat[51, 34],
                                     'cost': 129.36},
            'wood_landfill':        {'energy_input': df.iat[12, 35], 'energy_input_carbon': df.iat[16, 35],
                                     'op_carbon': df.iat[19, 35], 'energy_output': df.iat[23, 35],
                                     'energy_output_carbon_reduction': df.iat[27, 35],
                                     'resource_output': df.iat[41, 35], 'resource_output_carbon_reduction': df.iat[51, 35],
                                     'cost': 82.51},
            'yard_composting':      {'energy_input': df.iat[12, 37], 'energy_input_carbon': df.iat[16, 37],
                                     'op_carbon': df.iat[19, 37], 'energy_output': df.iat[23, 37],
                                     'energy_output_carbon_reduction': df.iat[27, 37],
                                     'resource_output': df.iat[41, 37], 'resource_output_carbon_reduction': df.iat[51, 37],
                                     'cost': 73.67},
            'yard_incineration':    {'energy_input': df.iat[12, 38], 'energy_input_carbon': df.iat[16, 38],
                                     'op_carbon': df.iat[19, 38], 'energy_output': df.iat[23, 38],
                                     'energy_output_carbon_reduction': df.iat[27, 38],
                                     'resource_output': df.iat[41, 38], 'resource_output_carbon_reduction': df.iat[51, 38],
                                     'cost': 129.36},
            'yard_landfill':        {'energy_input': df.iat[12, 39], 'energy_input_carbon': df.iat[16, 39],
                                     'op_carbon': df.iat[19, 39], 'energy_output': df.iat[23, 39],
                                     'energy_output_carbon_reduction': df.iat[27, 39],
                                     'resource_output': df.iat[41, 39], 'resource_output_carbon_reduction': df.iat[51, 39],
                                     'cost': 82.51},
            'other_landfill':       {'energy_input': df.iat[12, 41], 'energy_input_carbon': df.iat[16, 41],
                                     'op_carbon': df.iat[19, 41], 'energy_output': df.iat[23, 41],
                                     'energy_output_carbon_reduction': df.iat[27, 41],
                                     'resource_output': df.iat[41, 41], 'resource_output_carbon_reduction': df.iat[51, 41],
                                     'cost': 82.51},
        }
        transport_params = {
            'energy_per_ton_km': df.iat[12, 6],
            'carbon_per_ton_km': df.iat[16, 6] + df.iat[19, 6],
            'distance': 20,
            'cost': 170.91  
        }

        return disposal_params, transport_params
    
    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.episode_reward = 0.0
        comp_keys = self.composition_keys
        if options is not None and ('composition' in options) and ('total_mass' in options):
            comp = options['composition']
            comp_vals = np.array([float(comp[k]) for k in self.composition_keys], dtype=np.float32)
            comp_sum = np.sum(comp_vals)
            if comp_sum <= 0 or np.isnan(comp_sum):
                comp_vals = np.random.uniform(0.01, 1.0, size=9).astype(np.float32)
                comp_vals = comp_vals / np.sum(comp_vals)
                self.total_mass = float(np.random.uniform(0.5, 2.0))
            else:
                comp_vals = comp_vals / comp_sum
                self.total_mass = float(options['total_mass'])
            self.composition = {k: float(v) for k, v in zip(self.composition_keys, comp_vals)}
            self.composition_vector = comp_vals
        else:
            comp_vals = np.random.uniform(0.01, 1.0, size=9).astype(np.float32)
            comp_vals = comp_vals / np.sum(comp_vals)
            self.composition = {k: float(v) for k, v in zip(comp_keys, comp_vals)}
            self.composition_vector = comp_vals
            self.total_mass = float(np.random.uniform(0.5, 2.0))
        
        base = np.zeros(29, dtype=np.float32)
        
        base[0] = 1.0
        base[1] = 0.4
        base[2] = 0.3
        base[3:10] = np.array([0.15,0.6,0.05,0.05,0.05,0.1,0.0],dtype=np.float32)
        base[10:12] = np.array([0.25,0.75],dtype=np.float32)
        base[12:14] = np.array([0.34,0.66],dtype=np.float32)
        base[14:17] = np.array([0.68,0.06,0.26],dtype=np.float32)
        base[17:20] = np.array([0.1,0.2,0.7],dtype=np.float32)
        base[20:23] = np.array([0.18,0.27,0.55],dtype=np.float32)
        base[23:26] = np.array([0.17,0.16,0.67],dtype=np.float32)
        base[26:29] = np.array([0.63,0.07,0.3],dtype=np.float32)
        
        full = np.concatenate([base, self.composition_vector, [self.total_mass]]).astype(np.float32)
        self.state = full
        
        c, e, eo, cost, initial_resource_output = self.compute_metrics(base)
        self.initial_carbon = c
        self.prev_energy = e
        self.initial_cost = cost
        self.initial_energy_output = eo  
        self.prev_cost = cost
        self.initial_resource_output = initial_resource_output  
        assert self.state.shape == self.observation_space.shape, \
            f"state shape {self.state.shape} != obs_space {self.observation_space.shape}"
        info = {'composition': self.composition.copy(), 'total_mass': self.total_mass}
        # ==== Biomass-led cache init ====
        self.prev_violation = None
        self.ip1 = self.state[0]
        self.ip2 = self.state[1]
        self.ip4 = self.state[3:10].copy()
        self.ip11 = self.state[26:29].copy()
        self.prev_ip4 = self.ip4.copy()
        self.prev_ip11 = self.ip11.copy()
        self.no_progress_streak = 0
            # ===== SSP5-COST-SERVICE CONFIG =====
            
        self.SSP5C_COST_W = 200.0

        
        self.SSP5C_IP1_MIN = 1         
        self.SSP5C_COLLECT_PEN_W = 500.0   

        
        self.SSP5C_CLASS_W = 60.0          

        
        self.SSP5C_EOUT_LOW_W = 100.0   
        self.SSP5C_RES_LOW_W  = 100.0   

        
        self.SSP5C_CARBON_W = 0
        self.SSP5C_SLACK_W  = 0.0

        
        self.SSP5C_CAPS = {
            'cost_norm_cap':      5.0,
            'energy_out_norm_cap':5.0,
            'resource_norm_cap':  5.0,
        }
        # ====================================
    
    
        self.SSP4_CARBON_W = 300.0     
        self.SSP4_SLACK_W  = 100.0      

    
        self.SSP4_ENERGY_W   = 100.0   # total_energy_output
        self.SSP4_COST_W     = 200.0   
        self.SSP4_RESOURCE_W = 100.0   # total_resource_output

    
        self.SSP4_CAPS = {
            'energy_norm_cap':   5.0,
            'cost_norm_cap':     5.0,
            'resource_norm_cap': 5.0,
        }

        self.SSP4_ENERGY_INPUT_W = 0.0
        self.SSP4_CAPS.update({
            'energy_in_norm_cap': 5.0,    
        })
    # ============================================

        self.SSP3_CARBON_W = 300.0   
        self.SSP3_SLACK_W  = 100.0    

        
        self.SSP3_CAPS = {
            'collect_max': 1,   
            'classify_max': 1,  
            'recycle_max': 1,   
            'biotech_max': 1,   
        }
        
        self.SSP3_PENALTY = {
            'collect': 0,
            'classify': 10,
            'recycle': 10,
            'biotech': 10,
        }
        # ==== Incineration-led config ====
        
        
        self.INCIN_IP4_IDX = [0]   
        self.INCIN_IP7_IDX = [1]   
        self.INCIN_IP8_IDX = [1]   
        self.INCIN_IP9_IDX = [1]   
        self.INCIN_IP10_IDX = [1]  
        self.INCIN_IP11_IDX = [1]  

        
        
        self.comp_share = {k: 1.0 for k in self.composition_keys}

        
        self.prev_ip3 = None
        self.prev_ip4 = None
        self.prev_ip7 = None
        self.prev_ip8 = None
        self.prev_ip9 = None
        self.prev_ip10 = None
        self.prev_ip11 = None
        # ==== Incineration-led cache init ====
        self.ip3 = float(base[2])
        self.ip7 = base[14:17].copy()
        self.ip8 = base[17:20].copy()
        self.ip9 = base[20:23].copy()
        self.ip10 = base[23:26].copy()
        

        self.prev_ip3 = self.ip3
        self.prev_ip4 = self.ip4.copy()
        self.prev_ip7 = self.ip7.copy()
        self.prev_ip8 = self.ip8.copy()
        self.prev_ip9 = self.ip9.copy()
        self.prev_ip10 = self.ip10.copy()
        self.prev_ip11 = self.ip11.copy()

        
        self.comp_share = {k: float(self.composition[k]) for k in self.composition_keys}
        return self.state.copy(), info


    def step(self, action):
        
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print("action:", action)
            raise AssertionError("action contains NaN or Inf")
        
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = np.clip(action, -0.2, 0.2)
        
        new_state = self.state[:29] + action
        new_state = np.clip(new_state, 0.0, 1.0)
        
        for start, end in [(3,10), (10,12), (12,14), (14,17), (17,20), (20,23), (23,26), (26,29)]:
            sub_sum = np.sum(new_state[start:end])
            if sub_sum < 1e-8 or np.isnan(sub_sum):
                new_state[start:end] = 0
                new_state[start] = 1.0
            else:
                new_state[start:end] = new_state[start:end] / sub_sum
            new_state[start:end] = np.nan_to_num(new_state[start:end], nan=1.0, posinf=1.0, neginf=0.0)
        if np.any(np.isnan(new_state)) or np.any(np.isinf(new_state)):
            print("Warning: new_state contains NaN or Inf after normalization, fallback to previous state.")
            print("Action:", action)
            print("Previous state:", self.state)
            print("New state:", new_state)
            new_state = self.state[:29].copy()
        base = np.nan_to_num(new_state.astype(np.float32),
                             nan=0.0, posinf=0.0, neginf=0.0)
        full = np.concatenate([base, self.composition_vector, [self.total_mass]]).astype(np.float32)
        self.state = full
        if np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)):
            print("state:", self.state)
            raise AssertionError("state contains NaN or Inf")
        self.current_step += 1
        
        self.ip1  = float(new_state[0])
        self.ip2  = float(new_state[1])
        self.ip3  = float(new_state[2])
        self.ip4  = new_state[3:10].copy()
        self.ip5  = new_state[10:12].copy()
        self.ip6  = new_state[12:14].copy()
        self.ip7  = new_state[14:17].copy()
        self.ip8  = new_state[17:20].copy()
        self.ip9  = new_state[20:23].copy()
        self.ip10 = new_state[23:26].copy()
        self.ip11 = new_state[26:29].copy()
        
        total_carbon, total_energy_input, total_energy_output, total_cost, total_resource_output = self.compute_metrics(base)
        self.initial_energy_input = total_energy_input  
        total_carbon = np.nan_to_num(total_carbon, nan=0.0, posinf=0.0, neginf=0.0)
        total_energy_input = np.nan_to_num(total_energy_input, nan=0.0, posinf=0.0, neginf=0.0)
        total_energy_output = np.nan_to_num(total_energy_output, nan=0.0, posinf=0.0, neginf=0.0)
        total_cost = np.nan_to_num(total_cost, nan=0.0, posinf=0.0, neginf=0.0)
        total_resource_output = np.nan_to_num(total_resource_output, nan=0.0, posinf=0.0, neginf=0.0)

        
        if self.reward_mode == 'carbon':
            if total_carbon <= 0:
                ratio_neg = - total_carbon / self.initial_carbon
                reward = 100.0 + 50.0 * (ratio_neg ** 2)
            else:
                reward = 10.0 * (1 - (total_carbon / self.initial_carbon) ** 3)
        elif self.reward_mode == 'carbon_cost':
            EPS      = 1e-8
            MARGIN   = 0.0
            STEP_PEN = 400.0
            SLOPE    = 600.0
            POS_MAX  = 500.0
            CAP      = 5.0

            init_carbon = max(EPS, abs(self.initial_carbon))
            init_cost   = max(EPS,  self.initial_cost)

            if total_carbon > -MARGIN:
                violation = (total_carbon + MARGIN) / init_carbon
                reward = -STEP_PEN - SLOPE * violation
            else:
                cost_norm = total_cost / init_cost
                cost_norm = min(cost_norm, CAP)
                shaped = 1.0 / (1.0 + cost_norm)
                reward = POS_MAX * shaped
        elif self.reward_mode == 'carbon_energy':
            EPS = 1e-8
            ZERO_TOL = -500.0
            POS_REWARD = 100.0
            if total_carbon > 0.0:
                reward = ZERO_TOL
            else:
                en_norm = total_energy_output / (abs(self.initial_energy_output) + EPS)
                energy_bonus = 0.05 * POS_REWARD * en_norm
                reward = POS_REWARD + energy_bonus
        elif self.reward_mode == 'carbon_resource':
            
            EPS = 1e-8
            init_carbon   = max(EPS, abs(self.initial_carbon))
            init_resource = max(EPS, abs(self.initial_resource_output))

            
            buffer_frac = 0.02
            buffer = buffer_frac * init_carbon

            
            violation = max(0.0, (total_carbon + buffer) / init_carbon)
            penalty_lambda = 800.0
            penalty = penalty_lambda * violation

            
            cap = 5.0       
            pos_max = 500.0 
            base = 100.0    
            slack_w = 50.0  
            slack_cap = 1.0 

            res_norm = max(0.0, total_resource_output) / init_resource
            res_norm = min(res_norm, cap)
            res_shaped = res_norm / (1.0 + res_norm)  

            slack = max(0.0, (-total_carbon - buffer) / init_carbon)
            slack = min(slack, slack_cap)

            reward = base + pos_max * res_shaped + slack_w * slack - penalty
        elif self.reward_mode == 'carbon_biomass_led':
            EPS = 1e-8
            init_carbon = max(EPS, abs(self.initial_carbon))

            
            buffer_frac = 0.02
            buffer = buffer_frac * init_carbon

            
            violation = max(0.0, (total_carbon + buffer) / init_carbon)

            
            CAP = 5.0
            raw_biomass = max(0.0, self._biomass_score())
            biomass_norm = min(raw_biomass, CAP) / CAP  # ∈[0,1]

            
            prev_v = 1.0 if (self.prev_violation is None) else self.prev_violation
            progress = max(0.0, prev_v - violation)

            
            delta_bio = self._delta_biomass_pos()

            
            stagnation_eps = 1e-3            
            stagnation_patience = 3          
            if progress < stagnation_eps and biomass_norm >= 0.8:
                self.no_progress_streak += 1
            else:
                self.no_progress_streak = 0

            fallback_on = (self.no_progress_streak >= stagnation_patience)

            
            alpha_base, beta_base = 0.3, 0.7
            alpha = min(0.9, alpha_base + (0.4 if fallback_on else 0.0))
            beta = 1.0 - alpha

            
            lambda_v = 800.0   
            kappa    = 600.0   
            rho      = 30.0    

            if violation > 0.0:
                reward = - lambda_v * violation \
                         + kappa * progress * (alpha + beta * biomass_norm) \
                         + rho * delta_bio
            else:
                B = 100.0   
                T = 300.0   
                reward = B + T * biomass_norm
                terminated = True

            
            biomass_info = {
                'biomass_score': float(raw_biomass),
                'biomass_norm': float(biomass_norm),
                'violation': float(violation),
                'buffer': float(buffer),
                'fallback_on': bool(fallback_on),
            }

            self.prev_violation = violation
            self.prev_ip4 = self.ip4.copy()
            self.prev_ip11 = self.ip11.copy()
        elif self.reward_mode == 'carbon_incineration_led':
            EPS = 1e-8
            init_carbon = max(EPS, abs(self.initial_carbon))

            
            buffer_frac = 0.02
            buffer = buffer_frac * init_carbon

            
            violation = max(0.0, (total_carbon + buffer) / init_carbon)

            
            CAP = 5.0
            raw_incin = max(0.0, self._incineration_score())
            incin_norm = min(raw_incin, CAP) / CAP  # ∈[0,1]

            
            prev_v = 1.0 if (self.prev_violation is None) else self.prev_violation
            progress = max(0.0, prev_v - violation)

            
            delta_incin = self._delta_incineration_pos()

            
            stagnation_eps = 1e-3
            stagnation_patience = 3
            if progress < stagnation_eps and incin_norm >= 0.99:
                self.no_progress_streak += 1
            else:
                self.no_progress_streak = 0
            fallback_on = (self.no_progress_streak >= stagnation_patience)

            
            alpha_base, beta_base = 0.1, 0.9
            alpha = min(0.9, alpha_base + (0.2 if fallback_on else 0.0))
            beta = 1.0 - alpha

            
            lambda_v = 1000.0  
            kappa    = 100.0  
            rho      = 100.0   

            if violation > 0.0:
                reward = - lambda_v * violation \
                         + kappa * progress * (alpha + beta * incin_norm) \
                         + rho * delta_incin
            else:
                
                B = 400.0
                T = 80000.0
                reward = B + T * incin_norm
                terminated = True

            
            incin_info = {
                'incineration_score': float(raw_incin),
                'incineration_norm': float(incin_norm),
                'violation': float(violation),
                'buffer': float(buffer),
                'fallback_on': bool(fallback_on),
            }

            self.prev_violation = violation
            self.prev_ip3  = self.ip3
            self.prev_ip4  = self.ip4.copy()
            self.prev_ip7  = self.ip7.copy()
            self.prev_ip8  = self.ip8.copy()
            self.prev_ip9  = self.ip9.copy()
            self.prev_ip10 = self.ip10.copy()
            self.prev_ip11 = self.ip11.copy()
        elif self.reward_mode == 'ssp3_weighted_carbon':
            EPS = 1e-8
            init_carbon = max(EPS, abs(self.initial_carbon))

            
            pos_carbon   = max(0.0, total_carbon)
            carbon_norm  = pos_carbon / init_carbon
            carbon_score = 1.0 / (1.0 + carbon_norm)  
            slack        = min(1.0, max(0.0, -total_carbon / init_carbon))  
            carbon_reward = self.SSP3_CARBON_W * carbon_score + self.SSP3_SLACK_W * slack

            
            collect_use  = float(max(0.0, min(1.0, self.ip1)))           
            classify_use = float(max(0.0, min(1.0, self.ip2)))           
            recycle_use  = float(self._recycle_ratio_avg())              
            
            CAP = 5.0
            raw_bio     = max(0.0, self._biomass_score())
            biotech_use = min(raw_bio, CAP) / CAP

            def _hinge2(use, cap, weight):
                if cap <= EPS:
                    return 0.0
                exc = max(0.0, use - cap) / cap  
                return weight * (exc ** 2)       

            pen_collect  = _hinge2(collect_use,  self.SSP3_CAPS['collect_max'],  self.SSP3_PENALTY['collect'])
            pen_classify = _hinge2(classify_use, self.SSP3_CAPS['classify_max'], self.SSP3_PENALTY['classify'])
            pen_recycle  = _hinge2(recycle_use,  self.SSP3_CAPS['recycle_max'],  self.SSP3_PENALTY['recycle'])
            pen_biotech  = _hinge2(biotech_use,  self.SSP3_CAPS['biotech_max'],  self.SSP3_PENALTY['biotech'])
            penalty_total = pen_collect + pen_classify + pen_recycle + pen_biotech

            reward = carbon_reward - penalty_total

            
            ssp3_info = {
                'ssp3_carbon_score': float(carbon_score),
                'ssp3_slack': float(slack),
                'ssp3_collect_use': float(collect_use),
                'ssp3_classify_use': float(classify_use),
                'ssp3_recycle_use': float(recycle_use),
                'ssp3_biotech_use': float(biotech_use),
                'ssp3_penalty_total': float(penalty_total),
            }
        elif self.reward_mode == 'ssp4_weighted_multi':
            EPS = 1e-8
            
            init_carbon = max(EPS, abs(self.initial_carbon))
            pos_carbon   = max(0.0, total_carbon)
            carbon_norm  = pos_carbon / init_carbon
            carbon_score = 1.0 / (1.0 + carbon_norm)  
            slack        = min(1.0, max(0.0, -total_carbon / init_carbon))  
            carbon_reward = self.SSP4_CARBON_W * carbon_score + self.SSP4_SLACK_W * slack

            
            
            EN_CAP = float(self.SSP4_CAPS.get('energy_norm_cap',   5.0))
            CO_CAP = float(self.SSP4_CAPS.get('cost_norm_cap',     5.0))
            RE_CAP = float(self.SSP4_CAPS.get('resource_norm_cap', 5.0))

            EIN_CAP = float(self.SSP4_CAPS.get('energy_in_norm_cap', 5.0))

            
            ein_base = abs(getattr(self, 'initial_energy_input', 0.0)) + EPS
            ein_norm = (total_energy_input / ein_base) if 'total_energy_input' in locals() else (total_energy_input / ein_base)
            ein_norm = min(max(0.0, ein_norm), EIN_CAP)
            pen_ein  = max(0.0, self.SSP4_ENERGY_INPUT_W) * ein_norm

            
            en_norm = total_energy_output / (abs(self.initial_energy_output) + EPS)
            en_norm = min(max(0.0, en_norm), EN_CAP)
            S_en = en_norm / (1.0 + en_norm)

            
            cost_norm = total_cost / (self.initial_cost + EPS)
            cost_norm = min(max(0.0, cost_norm), CO_CAP)
            S_cost = 1.0 / (1.0 + cost_norm)

            
            res_norm = total_resource_output / (abs(self.initial_resource_output) + EPS)
            res_norm = min(max(0.0, res_norm), RE_CAP)
            S_res = res_norm / (1.0 + res_norm)

            
            reward = carbon_reward \
                   + self.SSP4_ENERGY_W   * S_en \
                   + self.SSP4_COST_W     * S_cost \
                   + self.SSP4_RESOURCE_W * S_res \
                   - pen_ein

            
            ssp4_info = {
                'ssp4_carbon_score': float(carbon_score),
                'ssp4_slack': float(slack),
                'energy_norm': float(en_norm), 'energy_score': float(S_en),
                'cost_norm': float(cost_norm), 'cost_score': float(S_cost),
                'resource_norm': float(res_norm), 'resource_score': float(S_res),
                'energy_input_norm': float(ein_norm),
                'energy_input_penalty': float(pen_ein),
            }
        elif self.reward_mode == 'ssp5_cost_service_led':
            EPS = 1e-8

            
            
            C_CAP = float(self.SSP5C_CAPS.get('cost_norm_cap', 5.0))
            cost_base = max(EPS, float(getattr(self, 'initial_cost', 0.0)))
            cost_norm = min(max(0.0, float(total_cost) / cost_base), C_CAP)
            S_cost = 1.0 / (1.0 + cost_norm)                 
            R_cost = self.SSP5C_COST_W * S_cost

            
            ip1 = float(np.clip(self.ip1, 0.0, 1.0))
            ip1_min = float(self.SSP5C_IP1_MIN)
            if ip1_min > EPS:
                gap = max(0.0, (ip1_min - ip1) / ip1_min)
                P_collect = self.SSP5C_COLLECT_PEN_W * (gap ** 2)
            else:
                P_collect = 0.0

            
            ip2 = float(np.clip(self.ip2, 0.0, 1.0))
            rec_share = float(self.composition['glass'] + self.composition['metal'] + 
                  self.composition['paper'] + self.composition['plastic'])
            R_class = self.SSP5C_CLASS_W * ip2 * (1.0 + 0.5 * rec_share)  


            
            
            ENO_CAP = float(self.SSP5C_CAPS.get('energy_out_norm_cap', 5.0))
            RES_CAP = float(self.SSP5C_CAPS.get('resource_norm_cap',   5.0))
            en_out_base = max(EPS, abs(float(getattr(self, 'initial_energy_output', 0.0))))
            res_out_base= max(EPS, abs(float(getattr(self, 'initial_resource_output', 0.0))))

            eout_norm = min(max(0.0, float(total_energy_output)   / en_out_base), ENO_CAP)
            res_norm  = min(max(0.0, float(total_resource_output) / res_out_base), RES_CAP)

            
            Q_eout = 1.0 / (1.0 + eout_norm)
            Q_res  = 1.0 / (1.0 + res_norm)

            R_eout = self.SSP5C_EOUT_LOW_W * Q_eout
            R_res  = self.SSP5C_RES_LOW_W  * Q_res

            
            R_carbon = 0.0

            
            reward = R_cost + R_class - P_collect + R_eout + R_res

            
            ssp5c_info = {
                'ip1': ip1, 'ip2': ip2,
                'cost_norm': float(cost_norm), 'cost_score': float(S_cost), 'cost_reward': float(R_cost),
                'collect_penalty': float(P_collect),
                'eout_norm': float(eout_norm), 'eout_score': float(Q_eout), 'eout_reward': float(R_eout),
                'res_norm':  float(res_norm),  'res_score':  float(Q_res),  'res_reward':  float(R_res),
                'reward_total': float(reward),
            }
        else:
            raise NotImplementedError(f"Reward mode {self.reward_mode} not implemented")

        self.episode_reward += reward
        # Preserve early termination from reward branch (if any)
        _early_done = locals().get('terminated', False)
        terminated = bool(_early_done) or (self.current_step >= self.max_steps)
        truncated = False  
        info = {
            "total_carbon": total_carbon,
            "total_energy_input": total_energy_input,
            "total_energy_output": total_energy_output,
            "total_cost": total_cost,
            "total_resource_output": total_resource_output
        }
        # Merge biomass-specific diagnostics (if present)
        if self.reward_mode == 'carbon_biomass_led':
            try:
                info.update(biomass_info)
            except NameError:
                pass
        if self.reward_mode == 'carbon_incineration_led':
            try:
                info.update(incin_info)
            except NameError:
                pass
        if self.reward_mode == 'ssp3_weighted_carbon':
            try:
                info.update(ssp3_info)
            except NameError:
                pass
        if self.reward_mode == 'ssp4_weighted_multi':
            try:
                info.update(ssp4_info)
            except NameError:
                pass
        if self.reward_mode == 'ssp5_cost_service_led':
            try:
                info.update(ssp5c_info)
            except NameError:
                pass
        if terminated:
            info['episode'] = {'r': self.episode_reward}
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(reward) or np.isinf(reward):
            print("reward:", reward)
            raise AssertionError("reward is NaN or Inf")
        if np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)):
            print("state (return):", self.state)
            raise AssertionError("state contains NaN or Inf (return)")
        assert self.state.shape == self.observation_space.shape, \
            f"state shape {self.state.shape} != obs_space {self.observation_space.shape}"
        return self.state.copy(), reward, terminated, truncated, info

    def compute_metrics(self, state):
        assert not np.any(np.isnan(state)), "compute_metrics: state contains NaN"
        
        
        controlled_ratio = state[0]              # IP1
        classification_ratio = state[1]          # IP2
        unsorted_incineration_ratio = state[2]   # IP3

        controlled_mass = controlled_ratio * self.total_mass
        uncontrolled_mass = self.total_mass - controlled_mass
        classified_mass = controlled_mass * classification_ratio
        unclassified_mass = controlled_mass * (1 - classification_ratio)

        unsorted_incineration_mass = unclassified_mass * unsorted_incineration_ratio
        unsorted_landfill_mass = unclassified_mass * (1 - unsorted_incineration_ratio)

        
        uncontrolled_burning_mass = uncontrolled_mass * 0.3
        uncontrolled_dumping_mass = uncontrolled_mass * 0.7

        
        food_mass    = classified_mass * self.composition['food']
        glass_mass   = classified_mass * self.composition['glass']
        metal_mass   = classified_mass * self.composition['metal']
        paper_mass   = classified_mass * self.composition['paper']
        plastic_mass = classified_mass * self.composition['plastic']
        rubber_mass  = classified_mass * self.composition['rubber']
        wood_mass    = classified_mass * self.composition['wood']
        yard_mass    = classified_mass * self.composition['yard']
        other_mass   = classified_mass * self.composition['other']  

        
        food_alloc = state[3:10]     
        glass_alloc = state[10:12]   
        metal_alloc = state[12:14]   
        paper_alloc = state[14:17]   
        plastic_alloc = state[17:20] 
        rubber_alloc = state[20:23]  
        wood_alloc = state[23:26]    
        yard_alloc = state[26:29]    

        
        food_methods = ['food_incineration', 'food_landfill', 'food_AD', 'food_composting',
                        'food_other_biotech', 'food_D&AF', 'food_sewer']
        glass_methods = ['glass_recycle', 'glass_landfill']
        metal_methods = ['metal_recycle', 'metal_landfill']
        paper_methods = ['paper_recycle', 'paper_incineration', 'paper_landfill']
        plastic_methods = ['plastic_recycle', 'plastic_incineration', 'plastic_landfill']
        rubber_methods = ['rubber_recycle', 'rubber_incineration', 'rubber_landfill']
        wood_methods = ['wood_recycle', 'wood_incineration', 'wood_landfill']
        yard_methods = ['yard_composting', 'yard_incineration', 'yard_landfill']

        
        food_masses = food_mass * food_alloc
        glass_masses = glass_mass * glass_alloc
        metal_masses = metal_mass * metal_alloc
        paper_masses = paper_mass * paper_alloc
        plastic_masses = plastic_mass * plastic_alloc
        rubber_masses = rubber_mass * rubber_alloc
        wood_masses = wood_mass * wood_alloc
        yard_masses = yard_mass * yard_alloc

        
        total_carbon = 0.0
        total_energy_input = 0.0
        total_energy_output = 0.0
        total_cost = 0.0
        total_resource_output = 0.0

        # uncontrolled system
        for method, mass in [('open_burning', uncontrolled_burning_mass), 
                             ('open_dumping', uncontrolled_dumping_mass)]:
            params = self.disposal_params[method]
            total_carbon += mass * (params['energy_input_carbon'] + params['op_carbon'])
            total_energy_input += mass * params['energy_input']
            total_energy_output += mass * params['energy_output']
            total_cost += mass * params['cost']
            total_resource_output += mass * params.get('resource_output', 0.0)

        # controlled system - unsorted
        for method, mass in [('unsorted_incineration', unsorted_incineration_mass), 
                             ('unsorted_landfill', unsorted_landfill_mass)]:
            params = self.disposal_params[method]
            total_carbon += mass * (params['energy_input_carbon'] + params['op_carbon'])
            total_energy_input += mass * params['energy_input']
            total_energy_output += mass * params['energy_output']
            total_cost += mass * self.transport_params['cost']
            total_carbon += mass * self.transport_distance * self.transport_carbon
            total_energy_input += mass * self.transport_distance * self.transport_energy
            total_carbon -= mass * (params['energy_output_carbon_reduction'] + params['resource_output_carbon_reduction'])
            total_cost += mass * params['cost']
            total_resource_output += mass * params.get('resource_output', 0.0)

        # controlled system - classified
        classified_routes = []
        for alloc_mass, method in zip(food_masses, food_methods):
            classified_routes.append((alloc_mass, method))
        for alloc_mass, method in zip(glass_masses, glass_methods):
            classified_routes.append((alloc_mass, method))
        for alloc_mass, method in zip(metal_masses, metal_methods):
            classified_routes.append((alloc_mass, method))
        for alloc_mass, method in zip(paper_masses, paper_methods):
            classified_routes.append((alloc_mass, method))
        for alloc_mass, method in zip(plastic_masses, plastic_methods):
            classified_routes.append((alloc_mass, method))
        for alloc_mass, method in zip(rubber_masses, rubber_methods):
            classified_routes.append((alloc_mass, method))
        for alloc_mass, method in zip(wood_masses, wood_methods):
            classified_routes.append((alloc_mass, method))
        for alloc_mass, method in zip(yard_masses, yard_methods):
            classified_routes.append((alloc_mass, method))
        
        classified_routes.append((other_mass, 'other_landfill'))
        
        for mass, method in classified_routes:
            params = self.disposal_params[method]
            total_carbon += mass * (params['energy_input_carbon'] + params['op_carbon'])
            total_energy_input += mass * params['energy_input']
            total_energy_output += mass * params['energy_output']
            total_cost += mass * self.transport_params['cost']
            total_carbon += mass * self.transport_distance * self.transport_carbon
            total_energy_input += mass * self.transport_distance * self.transport_energy
            total_carbon -= mass * (params['energy_output_carbon_reduction'] + params['resource_output_carbon_reduction'])
            total_cost += mass * params['cost']
            total_resource_output += mass * params.get('resource_output', 0.0)

        return total_carbon, total_energy_input, total_energy_output, total_cost, total_resource_output

    def _biomass_score(self) -> float:
        ip4_bio = float(np.sum(self.ip4[self.BIOMASS_IP4_IDX]))
        ip11_bio = float(np.sum(self.ip11[self.BIOMASS_IP11_IDX]))
        
        score_weighted = (self.ip1 * self.ip2 * self.food_share_for_weight) * ip4_bio \
                     + (self.ip1 * self.ip11_share_for_weight) * ip11_bio
        
        score_proxy = self.ip1 * (self.ip2 * ip4_bio + ip11_bio)
        use_weight = (self.food_share_for_weight != 1.0 or self.ip11_share_for_weight != 1.0)
        return score_weighted if use_weight else score_proxy

    def _delta_biomass_pos(self) -> float:
        d4 = np.clip(self.ip4[self.BIOMASS_IP4_IDX] - self.prev_ip4[self.BIOMASS_IP4_IDX], 0, None).sum()
        d11 = np.clip(self.ip11[self.BIOMASS_IP11_IDX] - self.prev_ip11[self.BIOMASS_IP11_IDX], 0, None).sum()
        return float(d4 + d11)

    def _incineration_score(self) -> float:
        ip1, ip2 = float(self.ip1), float(self.ip2)
        
        unsorted_term = ip1 * max(0.0, 1.0 - ip2) * max(0.0, self.ip3)

        
        w_food   = self.comp_share.get('food',   1.0)
        w_paper  = self.comp_share.get('paper',  1.0)
        w_plast  = self.comp_share.get('plastic',1.0)
        w_rubber = self.comp_share.get('rubber', 1.0)
        w_wood   = self.comp_share.get('wood',   1.0)
        w_yard   = self.comp_share.get('yard',   1.0)

        ip4_inc  = float(np.sum(self.ip4[self.INCIN_IP4_IDX]))
        ip7_inc  = float(np.sum(self.ip7[self.INCIN_IP7_IDX]))
        ip8_inc  = float(np.sum(self.ip8[self.INCIN_IP8_IDX]))
        ip9_inc  = float(np.sum(self.ip9[self.INCIN_IP9_IDX]))
        ip10_inc = float(np.sum(self.ip10[self.INCIN_IP10_IDX]))
        ip11_inc = float(np.sum(self.ip11[self.INCIN_IP11_IDX]))

        classified_term = ip1 * ip2 * (
            w_food  * ip4_inc  +
            w_paper * ip7_inc  +
            w_plast * ip8_inc  +
            w_rubber* ip9_inc  +
            w_wood  * ip10_inc +
            w_yard  * ip11_inc
        )
        return float(unsorted_term + classified_term)

    def _recycle_ratio_avg(self) -> float:
        vals = []
        if hasattr(self, 'ip5'):   vals.append(float(self.ip5[0]))   # glass recycle ratio
        if hasattr(self, 'ip6'):   vals.append(float(self.ip6[0]))   # metal recycle ratio
        if hasattr(self, 'ip7'):   vals.append(float(self.ip7[0]))   # paper recycle ratio
        if hasattr(self, 'ip8'):   vals.append(float(self.ip8[0]))   # plastic recycle ratio
        if hasattr(self, 'ip9'):   vals.append(float(self.ip9[0]))   # rubber recycle ratio
        if hasattr(self, 'ip10'):  vals.append(float(self.ip10[0]))  # wood recycle ratio
        if not vals:
            return 0.0
        v = float(np.mean(vals))
        return max(0.0, min(1.0, v))

    def _delta_incineration_pos(self) -> float:
        d_ip3  = max(0.0, self.ip3 - (self.prev_ip3 if self.prev_ip3 is not None else self.ip3))

        d_ip4  = np.clip(self.ip4[self.INCIN_IP4_IDX]  - self.prev_ip4[self.INCIN_IP4_IDX],   0, None).sum()
        d_ip7  = np.clip(self.ip7[self.INCIN_IP7_IDX]  - self.prev_ip7[self.INCIN_IP7_IDX],   0, None).sum()
        d_ip8  = np.clip(self.ip8[self.INCIN_IP8_IDX]  - self.prev_ip8[self.INCIN_IP8_IDX],   0, None).sum()
        d_ip9  = np.clip(self.ip9[self.INCIN_IP9_IDX]  - self.prev_ip9[self.INCIN_IP9_IDX],   0, None).sum()
        d_ip10 = np.clip(self.ip10[self.INCIN_IP10_IDX]- self.prev_ip10[self.INCIN_IP10_IDX], 0, None).sum()
        d_ip11 = np.clip(self.ip11[self.INCIN_IP11_IDX]- self.prev_ip11[self.INCIN_IP11_IDX], 0, None).sum()

        return float(d_ip3 + d_ip4 + d_ip7 + d_ip8 + d_ip9 + d_ip10 + d_ip11)
    
def print_intervention_scheme():
    print("IP1  - Controlled system 比例: state[0]")
    print("IP2  - 分类比例（controlled 内）: state[1]")
    print("IP3  - Unsorted incineration 比例（未分类部分）: state[2]")
    print("IP4  - Food waste 处理技术分配: state[3:10] (incineration, landfill, AD, composting, other biotech, D&AF, sewer)")
    print("IP5  - Glass waste 处理技术分配: state[10:12] (recycle, landfill)")
    print("IP6  - Metal waste 处理技术分配: state[12:14] (recycle, landfill)")
    print("IP7  - Paper waste 处理技术分配: state[14:17] (recycle, incineration, landfill)")
    print("IP8  - Plastic waste 处理技术分配: state[17:20] (recycle, incineration, landfill)")
    print("IP9  - Rubber waste 处理技术分配: state[20:23] (recycle, incineration, landfill)")
    print("IP10 - Wood waste 处理技术分配: state[23:26] (recycle, incineration, landfill)")
    print("IP11 - Yard waste 处理技术分配: state[26:29] (composting, incineration, landfill)")

def print_intervention_values(state):
    print("IP1 - Controlled system 比例: {:.4f}".format(state[0]))
    print("IP2 - 分类比例（controlled 内）: {:.4f}".format(state[1]))
    print("IP3 - Unsorted incineration 比例（未分类部分）: {:.4f}".format(state[2]))
    food_methods = ['food_incineration', 'food_landfill', 'food_AD', 'food_composting',
                    'food_other_biotech', 'food_D&AF', 'food_sewer']
    print("IP4 - Food waste 处理技术分配:")
    for i, method in enumerate(food_methods):
        print("   {}: {:.4f}".format(method, state[3+i]))
    glass_methods = ['glass_recycle', 'glass_landfill']
    print("IP5 - Glass waste 处理技术分配:")
    for i, method in enumerate(glass_methods):
        print("   {}: {:.4f}".format(method, state[10+i]))
    metal_methods = ['metal_recycle', 'metal_landfill']
    print("IP6 - Metal waste 处理技术分配:")
    for i, method in enumerate(metal_methods):
        print("   {}: {:.4f}".format(method, state[12+i]))
    paper_methods = ['paper_recycle', 'paper_incineration', 'paper_landfill']
    print("IP7 - Paper waste 处理技术分配:")
    for i, method in enumerate(paper_methods):
        print("   {}: {:.4f}".format(method, state[14+i]))
    plastic_methods = ['plastic_recycle', 'plastic_incineration', 'plastic_landfill']
    print("IP8 - Plastic waste 处理技术分配:")
    for i, method in enumerate(plastic_methods):
        print("   {}: {:.4f}".format(method, state[17+i]))
    rubber_methods = ['rubber_recycle', 'rubber_incineration', 'rubber_landfill']
    print("IP9 - Rubber waste 处理技术分配:")
    for i, method in enumerate(rubber_methods):
        print("   {}: {:.4f}".format(method, state[20+i]))
    wood_methods = ['wood_recycle', 'wood_incineration', 'wood_landfill']
    print("IP10 - Wood waste 处理技术分配:")
    for i, method in enumerate(wood_methods):
        print("   {}: {:.4f}".format(method, state[23+i]))
    yard_methods = ['yard_composting', 'yard_incineration', 'yard_landfill']
    print("IP11 - Yard waste 处理技术分配:")
    for i, method in enumerate(yard_methods):
        print("   {}: {:.4f}".format(method, state[26+i]))