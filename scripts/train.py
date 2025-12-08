import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from ENV import MSWEnv, print_intervention_scheme

class SaveMetricsCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveMetricsCallback, self).__init__(verbose)
        self.save_path = save_path
        # Initialize metrics containers
        self.metrics = {
            'timesteps': [],
            'ep_rew_mean': [],
            'approx_kl': [],
            'entropy_loss': [],
            'explained_variance': [],
            'loss': [],
            'policy_gradient_loss': [],
            'std': [],
            'value_loss': []
        }

    def _init_callback(self) -> None:
        self.ep_rewards = []

    def _on_step(self) -> bool:
        # Record episodic reward when available
        infos = self.locals.get('infos', [])
        if infos and 'episode' in infos[0]:
            self.ep_rewards.append(infos[0]['episode']['r'])
        return True

    def _on_rollout_end(self) -> None:
        # Record training metrics from logger
        self.metrics['timesteps'].append(self.num_timesteps)
        
        if self.ep_rewards:
            self.metrics['ep_rew_mean'].append(np.mean(self.ep_rewards))
            self.ep_rewards = []
        else:
            
            if self.metrics['ep_rew_mean']:
                self.metrics['ep_rew_mean'].append(self.metrics['ep_rew_mean'][-1])
            else:
                self.metrics['ep_rew_mean'].append(0.0)
        for key in ['approx_kl', 'entropy_loss', 'explained_variance', 'loss',
                    'policy_gradient_loss', 'std', 'value_loss']:
            self.metrics[key].append(self.model.logger.name_to_value.get(f'train/{key}', 0.0))

    def _on_training_end(self) -> None:
        # Ensure save directory
        os.makedirs(self.save_path, exist_ok=True)
        
        max_len = max(len(v) for v in self.metrics.values())
        for k, v in self.metrics.items():
            if len(v) < max_len:
                
                if v:
                    v.extend([v[-1]] * (max_len - len(v)))
                else:
                    v.extend([0.0] * (max_len - len(v)))
        # Save metrics to Excel
        df = pd.DataFrame(self.metrics)
        df.to_excel(os.path.join(self.save_path, 'training_metrics.xlsx'), index=False)
        # Plot each metric
        for metric, values in self.metrics.items():
            if metric == 'timesteps':
                continue
            plt.figure()
            plt.plot(self.metrics['timesteps'], values)
            plt.xlabel('Timesteps')
            plt.ylabel(metric)
            plt.title(f'{metric} over time')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f'{metric}.svg'), format='svg')
            plt.close()


if __name__ == '__main__':
    
    DATA_FILE = 'train_data.xlsx'        
    SAVE_DIR = 'training_results_ssp5_cost_service_led'
    TOTAL_TIMESTEPS = 200_000
    ENT_COEF = 0.1

    
    env = MSWEnv(
        data_file=DATA_FILE,
        reward_mode='ssp5_cost_service_led'
    )
    print('Loaded disposal_params and transport_params:')
    print(env.disposal_params)
    print(env.transport_params)

    
    callback = SaveMetricsCallback(save_path=SAVE_DIR, verbose=1)

    
    model = PPO('MlpPolicy', env, ent_coef=ENT_COEF, verbose=1)

    
    
    env._original_reset = env.reset
    def custom_reset(*args, **kwargs):
        obs, info = env._original_reset(*args, **kwargs)
        print(f"[采样] composition: {info['composition']}, total_mass: {info['total_mass']}")
        return obs, info
    env.reset = custom_reset
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    
    model.save(os.path.join(SAVE_DIR, 'ssp5_cost_service_led_agent.zip'))

    
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print('\n最终干预方案：')
    from ENV import print_intervention_values
    print_intervention_values(obs)
    print('\n测试信息：', info)
    print('累计奖励：', total_reward)
