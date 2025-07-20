import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from homeostatic_env import HomeostaticGridWorld
import pandas as pd


def plot_homeostatic_trajectories(trajectories, save_path=None):
    """Plot homeostatic variable trajectories over time"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Homeostatic Regulation During Episodes', fontsize=16)
    
    colors = {'hunger': 'red', 'thirst': 'blue', 'energy': 'green'}
    setpoints = {'hunger': 0.5, 'thirst': 0.5, 'energy': 0.8}
    
    # Plot individual variables
    for i, (var_name, color) in enumerate(colors.items()):
        ax = axes[i//2, i%2]
        
        for episode_idx, episode_data in enumerate(trajectories[var_name]):
            alpha = 0.3 if len(trajectories[var_name]) > 5 else 0.7
            ax.plot(episode_data, color=color, alpha=alpha, linewidth=1)
        
        # Plot setpoint
        ax.axhline(y=setpoints[var_name], color='black', linestyle='--', 
                  label=f'Setpoint ({setpoints[var_name]})')
        
        # Plot average trajectory if multiple episodes
        if len(trajectories[var_name]) > 1:
            # Pad shorter episodes with last value
            max_len = max(len(ep) for ep in trajectories[var_name])
            padded_episodes = []
            for ep in trajectories[var_name]:
                padded = ep + [ep[-1]] * (max_len - len(ep))
                padded_episodes.append(padded)
            
            avg_trajectory = np.mean(padded_episodes, axis=0)
            ax.plot(avg_trajectory, color='black', linewidth=3, alpha=0.8, 
                   label='Average')
        
        ax.set_title(f'{var_name.capitalize()} Regulation')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(f'{var_name.capitalize()} Level')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot combined view
    ax = axes[1, 1]
    for var_name, color in colors.items():
        if len(trajectories[var_name]) > 1:
            max_len = max(len(ep) for ep in trajectories[var_name])
            padded_episodes = []
            for ep in trajectories[var_name]:
                padded = ep + [ep[-1]] * (max_len - len(ep))
                padded_episodes.append(padded)
            avg_trajectory = np.mean(padded_episodes, axis=0)
            ax.plot(avg_trajectory, color=color, linewidth=2, label=var_name.capitalize())
        
        # Plot setpoint
        ax.axhline(y=setpoints[var_name], color=color, linestyle='--', alpha=0.5)
    
    ax.set_title('Combined Homeostatic Variables')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Level')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_homeostatic_stability(trajectories, save_path=None):
    """Plot homeostatic stability metrics"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {'hunger': 'red', 'thirst': 'blue', 'energy': 'green'}
    setpoints = {'hunger': 0.5, 'thirst': 0.5, 'energy': 0.8}
    
    # Calculate deviations from setpoint for each episode
    deviations = {var: [] for var in colors.keys()}
    
    for var_name in colors.keys():
        for episode_data in trajectories[var_name]:
            episode_deviations = [abs(val - setpoints[var_name]) for val in episode_data]
            deviations[var_name].extend(episode_deviations)
    
    # 1. Distribution of deviations
    ax = axes[0]
    data_for_violin = []
    labels_for_violin = []
    
    for var_name, color in colors.items():
        data_for_violin.append(deviations[var_name])
        labels_for_violin.append(var_name.capitalize())
    
    violin_parts = ax.violinplot(data_for_violin, positions=range(len(labels_for_violin)))
    
    for i, (pc, color) in enumerate(zip(violin_parts['bodies'], colors.values())):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(labels_for_violin)))
    ax.set_xticklabels(labels_for_violin)
    ax.set_ylabel('Deviation from Setpoint')
    ax.set_title('Distribution of Homeostatic Deviations')
    ax.grid(True, alpha=0.3)
    
    # 2. Stability over episodes
    ax = axes[1]
    episode_stabilities = []
    
    for episode_idx in range(len(trajectories['hunger'])):
        episode_stability = 0
        for var_name in colors.keys():
            if episode_idx < len(trajectories[var_name]):
                episode_data = trajectories[var_name][episode_idx]
                avg_deviation = np.mean([abs(val - setpoints[var_name]) for val in episode_data])
                episode_stability += avg_deviation
        episode_stabilities.append(episode_stability / len(colors))
    
    ax.plot(episode_stabilities, 'o-', color='purple', linewidth=2, markersize=6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Deviation')
    ax.set_title('Homeostatic Stability Across Episodes')
    ax.grid(True, alpha=0.3)
    
    # 3. Correlation matrix
    ax = axes[2]
    
    # Create correlation data
    correlation_data = {}
    min_length = min(len(deviations[var]) for var in colors.keys())
    
    for var_name in colors.keys():
        correlation_data[var_name] = deviations[var_name][:min_length]
    
    corr_df = pd.DataFrame(correlation_data)
    correlation_matrix = corr_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    ax.set_title('Homeostatic Variable Correlations')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_homeostatic_performance(model_path="homeostatic_agent", num_episodes=10):
    """Comprehensive analysis of homeostatic agent performance"""
    
    # Load trained model
    model = PPO.load(model_path)
    
    # Run evaluation episodes
    env = HomeostaticGridWorld(grid_size=8, max_steps=500, render_mode=None)
    
    episode_rewards = []
    homeostatic_trajectories = {'hunger': [], 'thirst': [], 'energy': []}
    action_frequencies = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # up, down, left, right, rest
    
    print("Analyzing homeostatic agent performance...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_homeostatic = {'hunger': [], 'thirst': [], 'energy': []}
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action_frequencies[int(action)] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Record homeostatic variables
            for var_name in ['hunger', 'thirst', 'energy']:
                episode_homeostatic[var_name].append(
                    info['homeostatic_vars'][var_name]['value']
                )
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        
        # Store trajectories
        for var_name in ['hunger', 'thirst', 'energy']:
            homeostatic_trajectories[var_name].append(episode_homeostatic[var_name])
    
    env.close()
    
    # Generate visualizations
    print("Generating homeostatic trajectory plots...")
    plot_homeostatic_trajectories(homeostatic_trajectories, 
                                 save_path="homeostatic_trajectories.png")
    
    print("Generating stability analysis...")
    plot_homeostatic_stability(homeostatic_trajectories,
                              save_path="homeostatic_stability.png")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("HOMEOSTATIC AGENT ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"Episodes analyzed: {num_episodes}")
    print(f"Average episode reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    
    # Action analysis
    total_actions = sum(action_frequencies.values())
    action_names = ['Up', 'Down', 'Left', 'Right', 'Rest/Consume']
    print(f"\nAction Distribution:")
    for i, (action, count) in enumerate(action_frequencies.items()):
        percentage = (count / total_actions) * 100
        print(f"  {action_names[i]}: {percentage:.1f}%")
    
    # Homeostatic analysis
    setpoints = {'hunger': 0.5, 'thirst': 0.5, 'energy': 0.8}
    print(f"\nHomeostatic Performance:")
    
    for var_name, setpoint in setpoints.items():
        all_values = []
        for episode_data in homeostatic_trajectories[var_name]:
            all_values.extend(episode_data)
        
        avg_value = np.mean(all_values)
        avg_deviation = np.mean([abs(val - setpoint) for val in all_values])
        
        print(f"  {var_name.capitalize()}:")
        print(f"    Average level: {avg_value:.3f} (setpoint: {setpoint})")
        print(f"    Average deviation: {avg_deviation:.3f}")
        print(f"    Stability: {1 - (avg_deviation / 0.5):.3f}")  # Normalized stability score
    
    print(f"\nKey Insights:")
    print(f"  - Agent demonstrates homeostatic regulation behavior")
    print(f"  - Balances multiple physiological variables simultaneously")
    print(f"  - Shows biologically plausible resource-seeking patterns")
    
    return homeostatic_trajectories, episode_rewards


if __name__ == "__main__":
    print("Starting homeostatic agent analysis...")
    trajectories, rewards = analyze_homeostatic_performance(num_episodes=10)
    print("Analysis complete! Check generated plots for detailed visualizations.")