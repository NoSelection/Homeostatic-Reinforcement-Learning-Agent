import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, Optional


class EasyHomeostaticGridWorld(gym.Env):
    """
    Easier version of Homeostatic RL environment with:
    - Slower decay rates (easier to survive)
    - More abundant resources 
    - Movement rewards (encourage exploration)
    - Smoother reward gradients
    - Better starting conditions
    """
    
    def __init__(self, 
                 grid_size: int = 8,
                 max_steps: int = 1000,
                 render_mode: Optional[str] = None):
        
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.current_step = 0
        
        # Define action space: up, down, left, right, rest
        self.action_space = spaces.Discrete(5)
        
        # Define observation space: position + homeostatic variables + resource info
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(8,),  # x, y, hunger, thirst, energy, nearest_food_dist, nearest_water_dist, nearest_rest_dist
            dtype=np.float32
        )
        
        # EASIER homeostatic variables (much slower decay)
        self.homeostatic_vars = {
            'hunger': {'value': 0.7, 'setpoint': 0.5, 'decay_rate': 0.003},  # 3x slower
            'thirst': {'value': 0.7, 'setpoint': 0.5, 'decay_rate': 0.004},  # 4x slower  
            'energy': {'value': 0.9, 'setpoint': 0.8, 'decay_rate': 0.002}   # 4x slower
        }
        
        # Environment elements
        self.agent_pos = np.array([grid_size//2, grid_size//2])  # Start in center
        self.food_sources = []
        self.water_sources = []
        self.rest_areas = []
        
        # Pygame for rendering
        self.window = None
        self.clock = None
        self.cell_size = 50
        
        self._generate_abundant_resources()
        
    def _generate_abundant_resources(self):
        """Generate MORE resources for easier learning"""
        # Clear existing resources
        self.food_sources = []
        self.water_sources = []
        self.rest_areas = []
        
        # More food sources (6 instead of 3)
        for _ in range(6):
            while True:
                pos = (np.random.randint(0, self.grid_size), 
                       np.random.randint(0, self.grid_size))
                if pos != tuple(self.agent_pos):
                    self.food_sources.append(pos)
                    break
        
        # More water sources (6 instead of 3)
        for _ in range(6):
            while True:
                pos = (np.random.randint(0, self.grid_size), 
                       np.random.randint(0, self.grid_size))
                if (pos != tuple(self.agent_pos) and 
                    pos not in self.food_sources):
                    self.water_sources.append(pos)
                    break
        
        # More rest areas (4 instead of 2)
        for _ in range(4):
            while True:
                pos = (np.random.randint(0, self.grid_size), 
                       np.random.randint(0, self.grid_size))
                if (pos != tuple(self.agent_pos) and 
                    pos not in self.food_sources and
                    pos not in self.water_sources):
                    self.rest_areas.append(pos)
                    break
    
    def _get_nearest_resource_distance(self, resource_list):
        """Get distance to nearest resource of given type"""
        if not resource_list:
            return 1.0  # Max distance if no resources
        
        min_dist = float('inf')
        for resource_pos in resource_list:
            dist = np.sqrt((self.agent_pos[0] - resource_pos[0])**2 + 
                          (self.agent_pos[1] - resource_pos[1])**2)
            min_dist = min(min_dist, dist)
        
        # Normalize to [0, 1]
        max_possible_dist = np.sqrt(2 * self.grid_size**2)
        return min_dist / max_possible_dist
    
    def _update_homeostasis(self, action: int):
        """Update homeostatic variables with BETTER reward shaping"""
        # Natural decay over time (much slower)
        for var_name, var_data in self.homeostatic_vars.items():
            var_data['value'] -= var_data['decay_rate']
            var_data['value'] = np.clip(var_data['value'], 0.0, 1.0)
        
        # Check if agent is at a resource and consuming it
        current_pos = tuple(self.agent_pos)
        
        # BETTER resource consumption (more forgiving)
        if current_pos in self.food_sources and action == 4:
            self.homeostatic_vars['hunger']['value'] = min(1.0, 
                self.homeostatic_vars['hunger']['value'] + 0.4)  # Bigger boost
        
        if current_pos in self.water_sources and action == 4:
            self.homeostatic_vars['thirst']['value'] = min(1.0,
                self.homeostatic_vars['thirst']['value'] + 0.5)  # Bigger boost
        
        if current_pos in self.rest_areas and action == 4:
            self.homeostatic_vars['energy']['value'] = min(1.0,
                self.homeostatic_vars['energy']['value'] + 0.3)  # Bigger boost
        
        # MUCH smaller movement cost (encourage exploration)
        if action in [0, 1, 2, 3]:
            self.homeostatic_vars['energy']['value'] -= 0.001  # Tiny cost
    
    def _calculate_shaped_reward(self) -> float:
        """Better reward function with multiple components"""
        
        # 1. Homeostatic reward (primary)
        homeostatic_reward = 0.0
        total_deviation = 0.0
        
        for var_name, var_data in self.homeostatic_vars.items():
            deviation = abs(var_data['value'] - var_data['setpoint'])
            total_deviation += deviation
            
            # Exponential penalty for getting close to death
            if var_data['value'] < 0.2:
                homeostatic_reward -= 10.0 * (0.2 - var_data['value'])
        
        homeostatic_reward -= total_deviation
        
        # 2. Exploration reward (encourage movement)
        exploration_reward = 0.0
        current_pos = tuple(self.agent_pos)
        
        # Small reward for being away from starting position
        start_pos = (self.grid_size//2, self.grid_size//2)
        dist_from_start = np.sqrt((current_pos[0] - start_pos[0])**2 + 
                                 (current_pos[1] - start_pos[1])**2)
        exploration_reward += 0.1 * min(dist_from_start / 3.0, 1.0)
        
        # 3. Resource proximity reward (guide towards resources when needed)
        proximity_reward = 0.0
        
        # If hungry, reward being close to food
        if self.homeostatic_vars['hunger']['value'] < 0.6:
            food_dist = self._get_nearest_resource_distance(self.food_sources)
            proximity_reward += 0.2 * (1.0 - food_dist)
        
        # If thirsty, reward being close to water
        if self.homeostatic_vars['thirst']['value'] < 0.6:
            water_dist = self._get_nearest_resource_distance(self.water_sources)
            proximity_reward += 0.2 * (1.0 - water_dist)
        
        # If tired, reward being close to rest
        if self.homeostatic_vars['energy']['value'] < 0.9:
            rest_dist = self._get_nearest_resource_distance(self.rest_areas)
            proximity_reward += 0.2 * (1.0 - rest_dist)
        
        # 4. Resource consumption reward
        consumption_reward = 0.0
        if current_pos in self.food_sources:
            consumption_reward += 1.0
        if current_pos in self.water_sources:
            consumption_reward += 1.0
        if current_pos in self.rest_areas:
            consumption_reward += 0.5
        
        # 5. Survival bonus
        survival_bonus = 0.1  # Small constant reward for staying alive
        
        total_reward = (homeostatic_reward + 
                       exploration_reward + 
                       proximity_reward + 
                       consumption_reward + 
                       survival_bonus)
        
        return total_reward
    
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
        
        # Calculate BETTER reward
        reward = self._calculate_shaped_reward()
        
        # Check termination conditions (more forgiving)
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Death condition: any homeostatic variable critically low (lower threshold)
        for var_data in self.homeostatic_vars.values():
            if var_data['value'] <= 0.0:
                terminated = True
                reward -= 20.0  # Death penalty
                break
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset agent position (center instead of corner)
        self.agent_pos = np.array([self.grid_size//2, self.grid_size//2])
        self.current_step = 0
        
        # Reset homeostatic variables (better starting values)
        self.homeostatic_vars['hunger']['value'] = 0.7
        self.homeostatic_vars['thirst']['value'] = 0.7  
        self.homeostatic_vars['energy']['value'] = 0.9
        
        # Regenerate abundant resources
        self._generate_abundant_resources()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_observation(self) -> np.ndarray:
        """Get enhanced observation with resource information"""
        obs = np.array([
            self.agent_pos[0] / self.grid_size,
            self.agent_pos[1] / self.grid_size,
            self.homeostatic_vars['hunger']['value'],
            self.homeostatic_vars['thirst']['value'],
            self.homeostatic_vars['energy']['value'],
            self._get_nearest_resource_distance(self.food_sources),
            self._get_nearest_resource_distance(self.water_sources),
            self._get_nearest_resource_distance(self.rest_areas)
        ], dtype=np.float32)
        return obs
    
    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            'homeostatic_vars': self.homeostatic_vars.copy(),
            'agent_pos': self.agent_pos.copy(),
            'step': self.current_step,
            'food_sources': len(self.food_sources),
            'water_sources': len(self.water_sources),
            'rest_areas': len(self.rest_areas)
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
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size + 120)
            )
            pygame.display.set_caption("Easy Homeostatic World")
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
        
        # Draw abundant resources
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
        
        # Draw agent (bigger and more visible)
        pygame.draw.circle(
            self.window, (255, 255, 0),  # Yellow for agent
            (self.agent_pos[0] * self.cell_size + self.cell_size // 2,
             self.agent_pos[1] * self.cell_size + self.cell_size // 2),
            self.cell_size // 3
        )
        
        # Draw homeostatic status with better colors
        font = pygame.font.Font(None, 24)
        y_offset = self.grid_size * self.cell_size + 10
        
        for i, (var_name, var_data) in enumerate(self.homeostatic_vars.items()):
            text = f"{var_name.capitalize()}: {var_data['value']:.2f}"
            # Better color coding
            if var_data['value'] > 0.5:
                color = (0, 150, 0)  # Dark green
            elif var_data['value'] > 0.3:
                color = (200, 200, 0)  # Yellow warning
            else:
                color = (200, 0, 0)  # Red danger
                
            text_surface = font.render(text, True, color)
            self.window.blit(text_surface, (10, y_offset + i * 25))
        
        # Show step counter
        step_text = font.render(f"Step: {self.current_step}/{self.max_steps}", True, (0, 0, 0))
        self.window.blit(step_text, (10, y_offset + 75))
        
        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS
    
    def close(self):
        """Clean up resources"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()