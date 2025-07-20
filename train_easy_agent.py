import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch
from easy_homeostatic_env import EasyHomeostaticGridWorld


class EasyHomeostaticCallback(BaseCallback):
    """Custom callback to track homeostatic variables during training"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.homeostatic_history = {
            'hunger': [],
            'thirst': [],
            'energy': []
        }
        self.episode_count = 0
        self.best_reward = -float('inf')
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        self.episode_count += 1
        
        # Log progress every 50 episodes
        if self.episode_count % 50 == 0:
            # Get recent performance
            if hasattr(self.training_env, 'get_attr'):
                try:
                    infos = self.training_env.get_attr('info')
                    if infos and len(infos) > 0:
                        recent_info = infos[0]
                        if 'homeostatic_vars' in recent_info:
                            vars_info = recent_info['homeostatic_vars']
                            print(f"Episode {self.episode_count}: Training progressing...")
                            print(f"  Hunger: {vars_info['hunger']['value']:.3f}")
                            print(f"  Thirst: {vars_info['thirst']['value']:.3f}") 
                            print(f"  Energy: {vars_info['energy']['value']:.3f}")
                except:
                    print(f"Episode {self.episode_count}: Training progressing...")


def create_easy_env():
    """Create the easier homeostatic environment"""
    return EasyHomeostaticGridWorld(grid_size=8, max_steps=500, render_mode=None)


def train_easy_agent(total_timesteps=200000):
    """Train PPO agent on easier homeostatic environment"""
    
    print("="*60)
    print("TRAINING EASY HOMEOSTATIC AGENT")
    print("="*60)
    print("Improvements in this version:")
    print("✓ 3-4x slower decay rates (easier survival)")
    print("✓ 2x more resources (6 food, 6 water, 4 rest)")
    print("✓ Better starting conditions (center spawn, higher initial values)")
    print("✓ Enhanced observations (distance to nearest resources)")
    print("✓ Shaped rewards (exploration + proximity + consumption bonuses)")
    print("✓ Minimal movement cost (encourage exploration)")
    print("✓ Better reward gradients (guide agent towards resources)")
    print()
    
    # Create vectorized environment
    env = make_vec_env(create_easy_env, n_envs=8)  # More parallel environments
    
    # Configure logging
    new_logger = configure("./easy_logs/", ["stdout", "csv", "tensorboard"])
    
    # Create PPO agent with better hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,      # Longer rollouts
        batch_size=128,    # Larger batches
        n_epochs=4,        # Fewer epochs per update (faster)
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,     # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[128, 128, 64],  # Larger network
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        tensorboard_log="./easy_logs/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Set the logger
    model.set_logger(new_logger)
    
    # Create callback
    callback = EasyHomeostaticCallback()
    
    print(f"Starting training for {total_timesteps} timesteps...")
    print("Expected improvements:")
    print("- Episode lengths should increase from ~34 to 200+")
    print("- Rewards should improve from ~-70 to positive values")
    print("- Agent should learn to seek resources proactively")
    print("- Movement patterns should emerge (exploration then resource seeking)")
    print()
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save the trained model
    model.save("easy_homeostatic_agent")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("Model saved as 'easy_homeostatic_agent'")
    print("Run evaluation to see the improved results!")
    
    return model, callback


def evaluate_easy_agent(model_path="easy_homeostatic_agent", num_episodes=10, render=False):
    """Evaluate the trained easy agent"""
    
    print("="*60)
    print("EVALUATING EASY HOMEOSTATIC AGENT")
    print("="*60)
    
    try:
        model = PPO.load(model_path)
    except:
        print(f"Could not load model from {model_path}")
        print("Please train the model first!")
        return None, None
    
    env = EasyHomeostaticGridWorld(grid_size=8, max_steps=500, 
                                  render_mode="human" if render else None)
    
    episode_rewards = []
    episode_lengths = []
    survival_count = 0
    
    homeostatic_trajectories = {
        'hunger': [],
        'thirst': [],
        'energy': []
    }
    
    print(f"Running {num_episodes} evaluation episodes...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_homeostatic = {'hunger': [], 'thirst': [], 'energy': []}
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Record homeostatic variables
            for var_name in ['hunger', 'thirst', 'energy']:
                episode_homeostatic[var_name].append(
                    info['homeostatic_vars'][var_name]['value']
                )
            
            if render:
                env.render()
            
            if terminated or truncated:
                break
        
        # Check if agent survived (didn't die from homeostatic failure)
        if not terminated:  # Only truncated (reached max steps)
            survival_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Store trajectories
        for var_name in ['hunger', 'thirst', 'energy']:
            homeostatic_trajectories[var_name].append(episode_homeostatic[var_name])
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Length = {episode_length}, "
              f"Survived = {'Yes' if not terminated else 'No'}")
    
    env.close()
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    survival_rate = survival_count / num_episodes
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average episode length: {avg_length:.1f} steps")
    print(f"Survival rate: {survival_rate:.1%} ({survival_count}/{num_episodes})")
    
    # Improvement analysis
    print(f"\nImprovement Analysis:")
    if avg_length > 100:
        print(f"✓ Episode length greatly improved: {avg_length:.1f} vs previous ~34")
    else:
        print(f"⚠ Episode length still short: {avg_length:.1f}")
        
    if avg_reward > -20:
        print(f"✓ Rewards significantly improved: {avg_reward:.2f} vs previous ~-70")
    else:
        print(f"⚠ Rewards still low: {avg_reward:.2f}")
        
    if survival_rate > 0.5:
        print(f"✓ Good survival rate: {survival_rate:.1%}")
    else:
        print(f"⚠ Low survival rate: {survival_rate:.1%}")
    
    return episode_rewards, homeostatic_trajectories


def quick_demo():
    """Quick demo of the easy environment with random actions"""
    print("="*60)
    print("EASY ENVIRONMENT DEMO")
    print("="*60)
    
    env = EasyHomeostaticGridWorld(grid_size=8, max_steps=100, render_mode="human")
    obs, info = env.reset()
    
    print("Showing easier environment with:")
    print("- More resources (6 food, 6 water, 4 rest)")
    print("- Slower decay rates")
    print("- Better starting position (center)")
    print("- Enhanced observations")
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if step % 20 == 0:
            print(f"Step {step}: Reward = {reward:.2f}")
            for var_name, var_data in info['homeostatic_vars'].items():
                print(f"  {var_name}: {var_data['value']:.3f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    env.close()


if __name__ == "__main__":
    print("Easy Homeostatic RL Training System")
    print("Select option:")
    print("1. Quick demo of easy environment")
    print("2. Train easy agent")
    print("3. Evaluate easy agent")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        quick_demo()
    elif choice == "2":
        timesteps = int(input("Enter training timesteps (default 200000): ") or "200000")
        train_easy_agent(total_timesteps=timesteps)
    elif choice == "3":
        render = input("Show visualization? (y/n): ").lower() == 'y'
        episodes = int(input("Number of episodes (default 10): ") or "10")
        evaluate_easy_agent(num_episodes=episodes, render=render)
    else:
        print("Invalid choice. Running demo...")
        quick_demo()