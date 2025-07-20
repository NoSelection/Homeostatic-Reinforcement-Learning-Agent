#!/usr/bin/env python3
"""
Homeostatic Reinforcement Learning Demo

This demonstrates a biologically-inspired RL agent that maintains internal 
physiological balance while navigating a grid world environment.

Key concepts:
- Homeostatic regulation: Agent maintains hunger, thirst, and energy levels
- Reward = negative deviation from physiological set-points  
- Emergent behavior: Agent learns to seek resources proactively
- Biological plausibility: Mimics real organisms' survival strategies
"""

import argparse
import os
from homeostatic_env import HomeostaticGridWorld
from train_agent import train_homeostatic_agent, evaluate_agent
from visualize_homeostasis import analyze_homeostatic_performance
from stable_baselines3 import PPO


def demo_environment():
    """Demonstrate the homeostatic environment with random actions"""
    print("="*60)
    print("HOMEOSTATIC ENVIRONMENT DEMO")
    print("="*60)
    print("Showing environment with random actions...")
    print("The agent has three homeostatic variables:")
    print("- Hunger (red circles = food)")
    print("- Thirst (blue circles = water)")  
    print("- Energy (green squares = rest areas)")
    print("Yellow circle = agent")
    print("\nPress any key to continue after viewing...")
    
    env = HomeostaticGridWorld(grid_size=6, max_steps=200, render_mode="human")
    obs, info = env.reset()
    
    for step in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    env.close()
    input("Press Enter to continue...")


def train_new_agent():
    """Train a new homeostatic agent"""
    print("="*60)
    print("TRAINING HOMEOSTATIC AGENT")
    print("="*60)
    
    timesteps = int(input("Enter training timesteps (default 30000): ") or "30000")
    
    print(f"Training agent for {timesteps} timesteps...")
    print("The agent will learn to:")
    print("- Maintain hunger around 0.5")
    print("- Maintain thirst around 0.5") 
    print("- Maintain energy around 0.8")
    print("- Minimize deviation from these set-points")
    
    model, callback = train_homeostatic_agent(total_timesteps=timesteps)
    print("Training complete! Model saved as 'homeostatic_agent'")
    
    return model


def run_trained_agent():
    """Run and visualize a trained agent"""
    if not os.path.exists("homeostatic_agent.zip"):
        print("No trained agent found! Please train one first.")
        return None
        
    print("="*60)
    print("EVALUATING TRAINED AGENT")
    print("="*60)
    
    model = PPO.load("homeostatic_agent")
    
    render = input("Show real-time visualization? (y/n): ").lower() == 'y'
    episodes = int(input("Number of episodes to run (default 5): ") or "5")
    
    print(f"Running {episodes} episodes...")
    rewards, trajectories = evaluate_agent(model, num_episodes=episodes, render=render)
    
    return model, rewards, trajectories


def analyze_performance():
    """Analyze agent performance with detailed visualizations"""
    if not os.path.exists("homeostatic_agent.zip"):
        print("No trained agent found! Please train one first.")
        return
        
    print("="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    episodes = int(input("Number of episodes to analyze (default 10): ") or "10")
    
    print("Generating comprehensive analysis...")
    analyze_homeostatic_performance(num_episodes=episodes)


def multi_agent_demo():
    """Demonstrate multi-agent homeostatic environment"""
    print("="*60)
    print("MULTI-AGENT EXTENSION (Coming Soon)")
    print("="*60)
    print("This would demonstrate:")
    print("- Multiple agents with shared resources")
    print("- Emergent cooperation/competition")
    print("- Resource scarcity effects")
    print("- Social homeostasis dynamics")
    print("\nImplementation placeholder - would require:")
    print("- Multi-agent environment wrapper")
    print("- Shared resource management")
    print("- Agent interaction protocols")


def main():
    parser = argparse.ArgumentParser(description="Homeostatic Reinforcement Learning Demo")
    parser.add_argument("--mode", choices=['demo', 'train', 'eval', 'analyze', 'multi'], 
                       help="Demo mode to run")
    
    args = parser.parse_args()
    
    if args.mode:
        mode = args.mode
    else:
        print("="*60)
        print("HOMEOSTATIC REINFORCEMENT LEARNING")
        print("="*60)
        print("A biologically-inspired RL agent that maintains physiological balance")
        print("\nAvailable demos:")
        print("1. Environment Demo - See the homeostatic world")
        print("2. Train Agent - Train a new homeostatic agent") 
        print("3. Evaluate Agent - Run trained agent")
        print("4. Analyze Performance - Detailed analysis with plots")
        print("5. Multi-Agent Preview - Future extension")
        
        choice = input("\nSelect option (1-5): ")
        mode_map = {'1': 'demo', '2': 'train', '3': 'eval', '4': 'analyze', '5': 'multi'}
        mode = mode_map.get(choice, 'demo')
    
    if mode == 'demo':
        demo_environment()
    elif mode == 'train':
        train_new_agent()
    elif mode == 'eval':
        run_trained_agent()
    elif mode == 'analyze':
        analyze_performance()
    elif mode == 'multi':
        multi_agent_demo()
    
    print("\n" + "="*60)
    print("Demo complete! Key insights about Homeostatic RL:")
    print("- Rewards based on physiological stability, not external goals")
    print("- Agent learns biologically plausible survival behaviors")
    print("- Demonstrates artificial consciousness via homeostasis")
    print("- Novel approach rarely seen in RL portfolios")
    print("="*60)


if __name__ == "__main__":
    main()