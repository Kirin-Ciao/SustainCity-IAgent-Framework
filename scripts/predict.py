import numpy as np
import copy
import os
import pandas as pd
from stable_baselines3 import PPO
from ENV import MSWEnv


REWARD_MODE = 'ssp4_weighted_multi'  
MODEL_PATH = os.path.join(f'training_results_{REWARD_MODE}', f'{REWARD_MODE}_agent.zip')
DATA_FILE = 'train_data.xlsx'
COUNTRY_DATA_FILE = 'country_data.xlsx'
SHEET_NAME = 'SSP4'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"未找到模型文件: {MODEL_PATH}")


df = pd.read_excel(COUNTRY_DATA_FILE, sheet_name=SHEET_NAME)
print('country_data列名:', list(df.columns))
col_map = {c.lower().replace(' ', '').replace('-', '').replace('_', ''): c for c in df.columns}
def get_col(name):
    key = name.lower().replace(' ', '').replace('-', '').replace('_', '')
    return col_map.get(key, name)
df_us = df[df[get_col('baseline_mode')] == 'CHN']

results = []
model = PPO.load(MODEL_PATH)


comp_cols = ['food','glass','metal','paper','plastic','rubber','wood','yard','other']
comp_median = df_us[comp_cols].median().to_dict()

def comp_nudge(action: np.ndarray, comp: dict, base_scale: float = 0.12) -> np.ndarray:
    a = action.astype(np.float32).copy()

    def z(key):  
        return float(comp[key]) - float(comp_median[key])

    def bump(idx, val):
        a[idx] += val

    s = base_scale  

    
    
    start = 3
    bump(start+2,  +1.00*s*z('food'))   # AD ↑
    bump(start+3,  +0.80*s*z('food'))   
    bump(start+0,  -0.60*s*z('food'))   
    bump(start+1,  -0.20*s*z('food'))   

    
    start = 10
    bump(start+0, +0.80*s*z('glass'));  bump(start+1, -0.80*s*z('glass'))

    
    start = 12
    bump(start+0, +0.80*s*z('metal'));  bump(start+1, -0.80*s*z('metal'))

    
    start = 14
    bump(start+0, +1.00*s*z('paper'))
    bump(start+1, -0.60*s*z('paper'))
    bump(start+2, -0.40*s*z('paper'))

    
    start = 17
    bump(start+0, +1.00*s*z('plastic'))
    bump(start+1, -0.50*s*z('plastic'))
    bump(start+2, -0.50*s*z('plastic'))

    
    start = 20
    bump(start+0, +0.60*s*z('rubber'))
    bump(start+1, -0.30*s*z('rubber'))
    bump(start+2, -0.30*s*z('rubber'))

    
    start = 23
    bump(start+0, +0.60*s*z('wood'))
    bump(start+1, -0.30*s*z('wood'))
    bump(start+2, -0.30*s*z('wood'))

    
    start = 26
    bump(start+0, +0.90*s*z('yard'))
    bump(start+1, -0.45*s*z('yard'))
    bump(start+2, -0.45*s*z('yard'))

    
    
    return np.clip(a, -0.2, 0.2)

for idx, row in df_us.iterrows():
    
    
    total_mass = row.get(get_col('total_mass'), row.get(get_col('total-sum'), None))
    if total_mass is None:
        print(f"警告: 行{idx}缺少total_mass/total-sum字段，跳过")
        continue
    composition = {
        'food': row[get_col('food')],
        'glass': row[get_col('glass')],
        'metal': row[get_col('metal')],
        'paper': row[get_col('paper')],
        'plastic': row[get_col('plastic')],
        'rubber': row[get_col('rubber')],
        'wood': row[get_col('wood')],
        'yard': row[get_col('yard')],
        'other': row[get_col('other')]
    }
    env = MSWEnv(DATA_FILE, reward_mode=REWARD_MODE)
    env.total_mass = total_mass
    env.composition = copy.deepcopy(composition)
    env.composition_vector = np.array([
        composition['food'], composition['glass'], composition['metal'],
        composition['paper'], composition['plastic'], composition['rubber'],
        composition['wood'], composition['yard'], composition['other']
    ], dtype=np.float32)
    obs, _ = env.reset(options={'composition': composition, 'total_mass': float(total_mass)})
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = comp_nudge(action, composition)  
        seed = hash((row[get_col('iso3c')], int(row[get_col('year')]))) & 0xffffffff
        rng = np.random.default_rng(seed)
        action = np.clip(action + rng.normal(0, 0.01, size=action.shape), -0.2, 0.2)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    def fmt(x):
        try:
            return round(float(x), 4)
        except:
            return x
    result = {
        'year': fmt(row[get_col('year')]),
        'iso3c': row[get_col('iso3c')],
        'region_id': row[get_col('region_id')],
        'country_name': row[get_col('country_name')],
        'income_id': row[get_col('income_id')],
        'baseline_mode': row[get_col('baseline_mode')],
        'total_mass': fmt(total_mass),
        'food': fmt(composition['food']),
        'glass': fmt(composition['glass']),
        'metal': fmt(composition['metal']),
        'paper': fmt(composition['paper']),
        'plastic': fmt(composition['plastic']),
        'rubber': fmt(composition['rubber']),
        'wood': fmt(composition['wood']),
        'yard': fmt(composition['yard']),
        'other': fmt(composition['other']),
        'IP1': fmt(obs[0]),
        'IP2': fmt(obs[1]),
        'IP3': fmt(obs[2]),
    }
    ip_ranges = {
        'IP4': (3, 10), 'IP5': (10, 12), 'IP6': (12, 14), 'IP7': (14, 17),
        'IP8': (17, 20), 'IP9': (20, 23), 'IP10': (23, 26), 'IP11': (26, 29)
    }
    for ip_name, (start, end) in ip_ranges.items():
        for i in range(start, end):
            result[f'{ip_name}_{i-start+1}'] = fmt(obs[i])
    
    result['total_carbon'] = fmt(info.get('total_carbon', None))
    result['total_energy_input'] = fmt(info.get('total_energy_input', None))
    result['total_energy_output'] = fmt(info.get('total_energy_output', None))
    result['total_cost'] = fmt(info.get('total_cost', None))
    result['total_resource_output'] = fmt(info.get('total_resource_output', None))
    result['total_reward'] = fmt(info.get('episode', {}).get('r', None))
    
    result['buffer'] = fmt(info.get('buffer', None))
    result['biomass_score'] = fmt(info.get('biomass_score', None))
    result['incineration_score'] = fmt(info.get('incineration_score', None))
    result['incineration_norm'] = fmt(info.get('incineration_norm', None))
    result['fallback_on'] = info.get('fallback_on', None)
    results.append(result)


output_file = f'country_predict_results_{REWARD_MODE}.xlsx'
results_df = pd.DataFrame(results)
results_df.to_excel(output_file, index=False)
print(f'预测完成，输入行数: {len(df_us)}, 输出行数: {len(results)}，结果已保存到 {output_file}')
