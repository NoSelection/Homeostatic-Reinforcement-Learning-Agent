import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, Optional


class HomeostaticGridWorld(gym.Env):
    """
    Homeostatic Reinforcement Learning environment where an agent must maintain
    internal physiological balance while navigating a grid world.
    """
    
    def __init__(self, 
                 grid_size: int = 10,
                 max_steps: int = 1000,
                 render_mode: Optional[str] = None):
        
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.current_step = 0
        
        # Define action space: up, down, left, right, rest
        self.action_space = spaces.Discrete(5)
        
        # Define observation space: position + homeostatic variables
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(5,),  # x, y, hunger, thirst, energy
            dtype=np.float32
        )
        
        # Homeostatic variables and their set-points
        self.homeostatic_vars = {
            'hunger': {'value': 0.5, 'setpoint': 0.5, 'decay_rate': 0.01},
            'thirst': {'value': 0.5, 'setpoint': 0.5, 'decay_rate': 0.015},
            'energy': {'value': 0.8, 'setpoint': 0.8, 'decay_rate': 0.008}
        }
        
        # Environment elements
        self.agent_pos = np.array([0, 0])
        self.food_sources = []
        self.water_sources = []
        self.rest_areas = []
        
        # Pygame for rendering
        self.window = None
        self.clock = None
        self.cell_size = 50
        
        self._generate_resources()
        
    def _generate_resources(self):
        """Generate food, water, and rest locations randomly"""
        # Ensure resources don't overlap
        all_positions = set()
        
        # Generate food sources
        for _ in range(3):
            while True:
                pos = (np.random.randint(0, self.grid_size), 
                       np.random.randint(0, self.grid_size))
                if pos not in all_positions and pos != (0, 0):
                    self.food_sources.append(pos)
                    all_positions.add(pos)
                    break
        
        # Generate water sources
        for _ in range(3):
            while True:
                pos = (np.random.randint(0, self.grid_size), 
                       np.random.randint(0, self.grid_size))
                if pos not in all_positions and pos != (0, 0):
                    self.water_sources.append(pos)
                    all_positions.add(pos)
                    break
        
        # Generate rest areas
        for _ in range(2):
            while True:
                pos = (np.random.randint(0, self.grid_size), 
                       np.random.randint(0, self.grid_size))
                if pos not in all_positions and pos != (0, 0):
                    self.rest_areas.append(pos)
                    all_positions.add(pos)
                    break
    
    def _update_homeostasis(self, action: int):
        """Update homeostatic variables based on action and time"""
        # Natural decay over time
        for var_name, var_data in self.homeostatic_vars.items():
            var_data['value'] -= var_data['decay_rate']
            var_data['value'] = np.clip(var_data['value'], 0.0, 1.0)
        
        # Check if agent is at a resource and consuming it
        current_pos = tuple(self.agent_pos)
        
        if current_pos in self.food_sources and action == 4:  # rest/consume action
            self.homeostatic_vars['hunger']['value'] = min(1.0, 
                self.homeostatic_vars['hunger']['value'] + 0.3)
        
        if current_pos in self.water_sources and action == 4:
            self.homeostatic_vars['thirst']['value'] = min(1.0,
                self.homeostatic_vars['thirst']['value'] + 0.4)
        
        if current_pos in self.rest_areas and action == 4:
            self.homeostatic_vars['energy']['value'] = min(1.0,
                self.homeostatic_vars['energy']['value'] + 0.2)
        
        # Movement costs energy
        if action in [0, 1, 2, 3]:  # movement actions
            self.homeostatic_vars['energy']['value'] -= 0.005
    
    def _calculate_homeostatic_reward(self) -> float:
        """Calculate reward based on homeostatic deviation from set-points"""
        total_deviation = 0.0
        
        for var_name, var_data in self.homeostatic_vars.items():
            deviation = abs(var_data['value'] - var_data['setpoint'])
            total_deviation += deviation
        
        # Reward is negative deviation (closer to set-point = higher reward)
        reward = -total_deviation
        
        # Severe penalty for critical states
        for var_name, var_data in self.homeostatic_vars.items():
            if var_data['value'] < 0.1:  # Critical low
                reward -= 5.0
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        # Move agent
        if action == 0:  # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        # action == 4 is rest/consume
        
        # Update homeostatic variables
        self._update_homeostasis(action)
        
        # Calculate reward
        reward = self._calculate_homeostatic_reward()
        
        # Check termination conditions
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Death condition: any homeostatic variable critically low
        for var_data in self.homeostatic_vars.values():
            if var_data['value'] <= 0.0:
                terminated = True
                reward -= 10.0  # Death penalty
                break
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset agent position
        self.agent_pos = np.array([0, 0])
        self.current_step = 0
        
        # Reset homeostatic variables
        for var_data in self.homeostatic_vars.values():
            var_data['value'] = var_data['setpoint']
        
        # Regenerate resources
        self.food_sources = []
        self.water_sources = []
        self.rest_areas = []
        self._generate_resources()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = np.array([
            self.agent_pos[0] / self.grid_size,
            self.agent_pos[1] / self.grid_size,
            self.homeostatic_vars['hunger']['value'],
            self.homeostatic_vars['thirst']['value'],
            self.homeostatic_vars['energy']['value']
        ], dtype=np.float32)
        return obs
    
    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            'homeostatic_vars': self.homeostatic_vars.copy(),
            'agent_pos': self.agent_pos.copy(),
            'step': self.current_step
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render for human viewing using pygame"""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size + 100)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Fill background
        self.window.fill((255, 255, 255))
        
        # Draw grid
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                self.window, (200, 200, 200),
                (x * self.cell_size, 0),
                (x * self.cell_size, self.grid_size * self.cell_size)
            )
        for y in range(self.grid_size + 1):
            pygame.draw.line(
                self.window, (200, 200, 200),
                (0, y * self.cell_size),
                (self.grid_size * self.cell_size, y * self.cell_size)
            )
        
        # Draw resources
        for pos in self.food_sources:
            pygame.draw.circle(
                self.window, (255, 0, 0),  # Red for food
                (pos[0] * self.cell_size + self.cell_size // 2,
                 pos[1] * self.cell_size + self.cell_size // 2),
                self.cell_size // 4
            )
        
        for pos in self.water_sources:
            pygame.draw.circle(
                self.window, (0, 0, 255),  # Blue for water
                (pos[0] * self.cell_size + self.cell_size // 2,
                 pos[1] * self.cell_size + self.cell_size // 2),
                self.cell_size // 4
            )
        
        for pos in self.rest_areas:
            pygame.draw.rect(
                self.window, (0, 255, 0),  # Green for rest
                (pos[0] * self.cell_size + self.cell_size // 4,
                 pos[1] * self.cell_size + self.cell_size // 4,
                 self.cell_size // 2, self.cell_size // 2)
            )
        
        # Draw agent
        pygame.draw.circle(
            self.window, (255, 255, 0),  # Yellow for agent
            (self.agent_pos[0] * self.cell_size + self.cell_size // 2,
             self.agent_pos[1] * self.cell_size + self.cell_size // 2),
            self.cell_size // 3
        )
        
        # Draw homeostatic status
        font = pygame.font.Font(None, 24)
        y_offset = self.grid_size * self.cell_size + 10
        
        for i, (var_name, var_data) in enumerate(self.homeostatic_vars.items()):
            text = f"{var_name.capitalize()}: {var_data['value']:.2f}"
            color = (0, 255, 0) if var_data['value'] > 0.3 else (255, 0, 0)
            text_surface = font.render(text, True, color)
            self.window.blit(text_surface, (10, y_offset + i * 25))
        
        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS
    
    def close(self):
        """Clean up resources"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()