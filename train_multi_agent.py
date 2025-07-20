import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
from multi_agent_env import MultiAgentHomeostaticWorld


class MultiAgentWrapper(gym.Env):
    """Wrapper to make multi-agent environment compatible with single-agent algorithms"""
    
    def __init__(self, env):
        self.env = env
        self.num_agents = env.num_agents
        
        # Create flattened action and observation spaces
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
        self.agent_rewards = []
        self.cooperation_history = []
        self.conflict_history = []
    
    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        return observations[0], infos[0]  # Return first agent's obs/info
    
    def step(self, action):
        # Duplicate action for all agents (or use independent policies)
        actions = [action] * self.num_agents
        
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        
        # Store metrics
        self.agent_rewards.append(rewards)
        if infos:
            self.cooperation_history.append(infos[0].get('cooperation_events', 0))
            self.conflict_history.append(infos[0].get('resource_conflicts', 0))
        
        # Return aggregated results
        return (observations[0], 
                np.mean(rewards), 
                any(terminated), 
                any(truncated), 
                infos[0])
    
    def render(self, mode='human'):
        return self.env.render()
    
    def close(self):
        return self.env.close()


class MultiAgentCallback(BaseCallback):
    """Callback to track multi-agent specific metrics"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.cooperation_scores = []
        self.survival_rates = []
        self.resource_efficiency = []
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        self.episode_count += 1
        
        if self.episode_count % 20 == 0:
            print(f"Multi-agent episode {self.episode_count}: Analyzing cooperation dynamics...")


def create_multi_agent_env(sharing_enabled=True, resource_scarcity=0.7):
    """Create multi-agent homeostatic environment"""
    env = MultiAgentHomeostaticWorld(
        grid_size=10,
        num_agents=3,
        max_steps=300,
        resource_scarcity=resource_scarcity,
        sharing_enabled=sharing_enabled,
        render_mode=None
    )
    return MultiAgentWrapper(env)


def train_multi_agent_comparison():
    """Train agents in both cooperative and competitive scenarios"""
    
    print("="*60)
    print("MULTI-AGENT HOMEOSTATIC LEARNING EXPERIMENT")
    print("="*60)
    print("Training scenarios:")
    print("1. Cooperative (resource sharing enabled)")
    print("2. Competitive (resource sharing disabled)")
    print("3. Resource scarcity comparison")
    
    scenarios = [
        {"name": "Cooperative", "sharing": True, "scarcity": 0.7},
        {"name": "Competitive", "sharing": False, "scarcity": 0.7},
        {"name": "Scarce Resources", "sharing": True, "scarcity": 0.4}
    ]
    
    models = {}
    metrics = {}
    
    for scenario in scenarios:
        print(f"\nTraining {scenario['name']} scenario...")
        
        # Create environment
        env_func = lambda: create_multi_agent_env(
            sharing_enabled=scenario['sharing'],
            resource_scarcity=scenario['scarcity']
        )
        env = DummyVecEnv([env_func])
        
        # Create model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=32,
            n_epochs=10,
            gamma=0.95,
            policy_kwargs=dict(
                net_arch=[128, 128],
                activation_fn=torch.nn.ReLU
            ),
            verbose=1
        )
        
        # Train
        callback = MultiAgentCallback()
        model.learn(total_timesteps=30000, callback=callback, progress_bar=True)
        
        # Save model
        model_name = f"multi_agent_{scenario['name'].lower()}"
        model.save(model_name)
        models[scenario['name']] = model
        
        print(f"{scenario['name']} training complete!")
    
    return models


def evaluate_multi_agent_scenarios(models=None, num_episodes=10):
    """Evaluate and compare different multi-agent scenarios"""
    
    if models is None:
        # Load pre-trained models
        models = {}
        scenario_names = ["Cooperative", "Competitive", "Scarce Resources"]
        for name in scenario_names:
            try:
                model_name = f"multi_agent_{name.lower().replace(' ', '_')}"
                models[name] = PPO.load(model_name)
            except:
                print(f"Model for {name} not found. Please train first.")
                return
    
    scenarios = [
        {"name": "Cooperative", "sharing": True, "scarcity": 0.7},
        {"name": "Competitive", "sharing": False, "scarcity": 0.7},
        {"name": "Scarce Resources", "sharing": True, "scarcity": 0.4}
    ]
    
    results = {}
    
    print("="*60)
    print("MULTI-AGENT SCENARIO COMPARISON")
    print("="*60)
    
    for scenario in scenarios:
        print(f"\nEvaluating {scenario['name']} scenario...")
        
        # Create environment
        base_env = MultiAgentHomeostaticWorld(
            grid_size=10,
            num_agents=3,
            max_steps=300,
            resource_scarcity=scenario['scarcity'],
            sharing_enabled=scenario['sharing'],
            render_mode=None
        )
        
        model = models[scenario['name']]
        
        episode_metrics = {
            'rewards': [],
            'survival_rates': [],
            'cooperation_events': [],
            'resource_conflicts': [],
            'agent_lifespans': []
        }
        
        for episode in range(num_episodes):
            observations, infos = base_env.reset()
            episode_rewards = [0] * base_env.num_agents
            episode_steps = 0
            agent_death_steps = [None] * base_env.num_agents
            
            while episode_steps < base_env.max_steps:
                # Get actions for all agents (simplified: same policy for all)
                actions = []
                for i in range(base_env.num_agents):
                    if base_env.agents[i]['alive']:
                        action, _ = model.predict(observations[i], deterministic=True)
                        actions.append(action)
                    else:
                        actions.append(0)  # Dead agents don't act
                
                observations, rewards, terminated, truncated, infos = base_env.step(actions)
                
                # Track metrics
                for i, reward in enumerate(rewards):
                    episode_rewards[i] += reward
                    if terminated[i] and agent_death_steps[i] is None:
                        agent_death_steps[i] = episode_steps
                
                episode_steps += 1
                
                if all(terminated) or any(truncated):
                    break
            
            # Calculate episode metrics
            alive_agents = sum(1 for agent in base_env.agents if agent['alive'])
            survival_rate = alive_agents / base_env.num_agents
            
            episode_metrics['rewards'].append(np.mean(episode_rewards))
            episode_metrics['survival_rates'].append(survival_rate)
            episode_metrics['cooperation_events'].append(infos[0]['cooperation_events'])
            episode_metrics['resource_conflicts'].append(infos[0]['resource_conflicts'])
            
            # Agent lifespans
            lifespans = []
            for i, death_step in enumerate(agent_death_steps):
                if death_step is None:
                    lifespans.append(episode_steps)  # Survived full episode
                else:
                    lifespans.append(death_step)
            episode_metrics['agent_lifespans'].append(np.mean(lifespans))
        
        # Calculate averages
        avg_metrics = {key: np.mean(values) for key, values in episode_metrics.items()}
        avg_metrics['std_rewards'] = np.std(episode_metrics['rewards'])
        
        results[scenario['name']] = avg_metrics
        
        # Print results
        print(f"  Average reward: {avg_metrics['rewards']:.2f} ± {avg_metrics['std_rewards']:.2f}")
        print(f"  Survival rate: {avg_metrics['survival_rates']:.1%}")
        print(f"  Cooperation events: {avg_metrics['cooperation_events']:.1f}")
        print(f"  Resource conflicts: {avg_metrics['resource_conflicts']:.1f}")
        print(f"  Average lifespan: {avg_metrics['agent_lifespans']:.1f} steps")
    
    # Generate comparison visualization
    visualize_multi_agent_results(results)
    
    return results


def visualize_multi_agent_results(results):
    """Create visualizations comparing multi-agent scenarios"""
    
    scenarios = list(results.keys())
    metrics = ['rewards', 'survival_rates', 'cooperation_events', 'resource_conflicts']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Agent Homeostatic Learning: Scenario Comparison', fontsize=16)
    
    colors = ['#2E8B57', '#DC143C', '#4169E1']  # Green, Red, Blue
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        values = [results[scenario][metric] for scenario in scenarios]
        bars = ax.bar(scenarios, values, color=colors)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('multi_agent_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create cooperation vs competition analysis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    cooperation_data = results['Cooperative']
    competition_data = results['Competitive']
    
    metrics_to_compare = ['survival_rates', 'cooperation_events', 'resource_conflicts']
    x_pos = np.arange(len(metrics_to_compare))
    width = 0.35
    
    coop_values = [cooperation_data[m] for m in metrics_to_compare]
    comp_values = [competition_data[m] for m in metrics_to_compare]
    
    bars1 = ax.bar(x_pos - width/2, coop_values, width, label='Cooperative', color='#2E8B57')
    bars2 = ax.bar(x_pos + width/2, comp_values, width, label='Competitive', color='#DC143C')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Cooperation vs Competition in Multi-Agent Homeostasis')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_compare])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('cooperation_vs_competition.png', dpi=300, bbox_inches='tight')
    plt.show()


def demo_multi_agent_environment():
    """Interactive demo of multi-agent homeostatic environment"""
    
    print("="*60)
    print("MULTI-AGENT HOMEOSTATIC ENVIRONMENT DEMO")
    print("="*60)
    
    sharing = input("Enable resource sharing? (y/n): ").lower() == 'y'
    scarcity = float(input("Resource scarcity (0.4-1.0, lower=scarcer): ") or "0.7")
    
    env = MultiAgentHomeostaticWorld(
        grid_size=8,
        num_agents=3,
        max_steps=200,
        resource_scarcity=scarcity,
        sharing_enabled=sharing,
        render_mode="human"
    )
    
    print(f"\nDemo starting with:")
    print(f"- Resource sharing: {'Enabled' if sharing else 'Disabled'}")
    print(f"- Resource scarcity: {scarcity}")
    print(f"- 3 agents competing/cooperating for survival")
    print("\nWatch how agents interact with shared resources!")
    
    observations, infos = env.reset()
    
    for step in range(200):
        # Random actions for demo
        actions = [env.action_space.sample()[i] for i in range(env.num_agents)]
        
        observations, rewards, terminated, truncated, infos = env.step(actions)
        env.render()
        
        if all(terminated) or any(truncated):
            break
    
    env.close()
    
    # Print final stats
    print(f"\nDemo completed!")
    print(f"Final cooperation events: {infos[0]['cooperation_events']}")
    print(f"Final resource conflicts: {infos[0]['resource_conflicts']}")
    
    alive_agents = sum(1 for agent in env.agents if agent['alive'])
    print(f"Agents survived: {alive_agents}/{env.num_agents}")


if __name__ == "__main__":
    import gymnasium as gym
    
    print("Multi-Agent Homeostatic Learning System")
    print("Select option:")
    print("1. Demo environment")
    print("2. Train scenarios") 
    print("3. Evaluate scenarios")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        demo_multi_agent_environment()
    elif choice == "2":
        models = train_multi_agent_comparison()
        print("Training complete! Run evaluation to see results.")
    elif choice == "3":
        results = evaluate_multi_agent_scenarios()
        print("Evaluation complete! Check generated plots.")
    else:
        print("Running demo by default...")
        demo_multi_agent_environment()