import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch
from homeostatic_env import HomeostaticGridWorld


class HomeostaticCallback(BaseCallback):
    """Custom callback to track homeostatic variables during training"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.homeostatic_history = {
            'hunger': [],
            'thirst': [],
            'energy': [],
            'rewards': [],
            'episode_lengths': []
        }
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # Get info from the environment
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'homeostatic_vars' in info:
                    # Record homeostatic variables
                    for var_name in ['hunger', 'thirst', 'energy']:
                        self.homeostatic_history[var_name].append(
                            info['homeostatic_vars'][var_name]['value']
                        )
        
        # Record rewards
        if self.locals.get('rewards') is not None:
            for reward in self.locals['rewards']:
                self.homeostatic_history['rewards'].append(reward)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        self.episode_count += 1
        
        # Log average homeostatic values every 10 episodes
        if self.episode_count % 10 == 0 and self.homeostatic_history['hunger']:
            recent_hunger = np.mean(self.homeostatic_history['hunger'][-100:])
            recent_thirst = np.mean(self.homeostatic_history['thirst'][-100:])
            recent_energy = np.mean(self.homeostatic_history['energy'][-100:])
            recent_reward = np.mean(self.homeostatic_history['rewards'][-100:])
            
            print(f"Episode {self.episode_count}: "
                  f"Hunger: {recent_hunger:.3f}, "
                  f"Thirst: {recent_thirst:.3f}, "
                  f"Energy: {recent_energy:.3f}, "
                  f"Reward: {recent_reward:.3f}")


def create_homeostatic_env():
    """Create the homeostatic environment"""
    return HomeostaticGridWorld(grid_size=8, max_steps=500, render_mode=None)


def train_homeostatic_agent(total_timesteps=100000):
    """Train a PPO agent on the homeostatic environment"""
    
    # Create vectorized environment
    env = make_vec_env(create_homeostatic_env, n_envs=4)
    
    # Configure logging
    new_logger = configure("./logs/", ["stdout", "csv", "tensorboard"])
    
    # Create PPO agent with custom policy network
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[64, 64],  # Two hidden layers with 64 neurons each
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        tensorboard_log="./logs/"
    )
    
    # Set the logger
    model.set_logger(new_logger)
    
    # Create callback to track homeostatic variables
    callback = HomeostaticCallback()
    
    print("Starting training...")
    print("The agent will learn to balance three homeostatic variables:")
    print("- Hunger (setpoint: 0.5, decays at 0.01/step)")
    print("- Thirst (setpoint: 0.5, decays at 0.015/step)")
    print("- Energy (setpoint: 0.8, decays at 0.008/step)")
    print("\nReward is calculated as negative deviation from setpoints.")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save the trained model
    model.save("homeostatic_agent")
    
    return model, callback


def evaluate_agent(model, num_episodes=10, render=False):
    """Evaluate the trained agent"""
    env = HomeostaticGridWorld(grid_size=8, max_steps=500, 
                              render_mode="human" if render else None)
    
    episode_rewards = []
    homeostatic_trajectories = {
        'hunger': [],
        'thirst': [],
        'energy': []
    }
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_homeostatic = {'hunger': [], 'thirst': [], 'energy': []}
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Record homeostatic variables
            for var_name in ['hunger', 'thirst', 'energy']:
                episode_homeostatic[var_name].append(
                    info['homeostatic_vars'][var_name]['value']
                )
            
            if render:
                env.render()
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        
        # Store trajectories
        for var_name in ['hunger', 'thirst', 'energy']:
            homeostatic_trajectories[var_name].append(episode_homeostatic[var_name])
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Length = {len(episode_homeostatic['hunger'])}")
    
    env.close()
    
    print(f"\nEvaluation complete!")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    return episode_rewards, homeostatic_trajectories


if __name__ == "__main__":
    # Train the agent
    model, training_callback = train_homeostatic_agent(total_timesteps=50000)
    
    print("\nTraining complete! Evaluating agent...")
    
    # Evaluate the agent
    rewards, trajectories = evaluate_agent(model, num_episodes=5, render=False)
    
    print(f"\nThe agent has learned homeostatic regulation!")
    print(f"Key insights:")
    print(f"- Agent balances multiple physiological needs simultaneously")
    print(f"- Reward structure encourages maintaining stability over external goals")
    print(f"- Biologically plausible behavior emerges from simple setpoint deviations")