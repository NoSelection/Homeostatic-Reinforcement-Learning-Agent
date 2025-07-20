import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Dict, Optional
import pygame

class MultiAgentHomeostaticWorld(gym.Env):
    """Simple multi-agent extension of the homeostatic grid world.

    Agents share resources and must maintain hunger, thirst and energy.
    Cooperation events are counted when agents consume the same resource
    simultaneously if sharing is enabled. If sharing is disabled, such
    encounters count as resource conflicts.
    """

    def __init__(self,
                 grid_size: int = 8,
                 num_agents: int = 2,
                 max_steps: int = 300,
                 resource_scarcity: float = 1.0,
                 sharing_enabled: bool = True,
                 render_mode: Optional[str] = None):
        super().__init__()

        self.grid_size = grid_size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.resource_scarcity = np.clip(resource_scarcity, 0.1, 1.0)
        self.sharing_enabled = sharing_enabled

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

        self.agents = []
        self.food_sources: List[Tuple[int, int]] = []
        self.water_sources: List[Tuple[int, int]] = []
        self.rest_areas: List[Tuple[int, int]] = []

        self.current_step = 0
        self.window = None
        self.clock = None
        self.cell_size = 40

        self._generate_resources()

    # ------------------------------------------------------------------
    # Environment helper methods
    # ------------------------------------------------------------------
    def _generate_resources(self) -> None:
        """Generate grid positions for resources based on scarcity."""
        self.food_sources = []
        self.water_sources = []
        self.rest_areas = []
        all_pos = set()

        def add_resources(target: List[Tuple[int, int]], count: int):
            for _ in range(count):
                while True:
                    pos = (np.random.randint(0, self.grid_size),
                           np.random.randint(0, self.grid_size))
                    if pos not in all_pos:
                        target.append(pos)
                        all_pos.add(pos)
                        break

        base_food = max(1, int(6 * self.resource_scarcity))
        base_water = max(1, int(6 * self.resource_scarcity))
        base_rest = max(1, int(4 * self.resource_scarcity))

        add_resources(self.food_sources, base_food)
        add_resources(self.water_sources, base_water)
        add_resources(self.rest_areas, base_rest)

    def _init_agents(self):
        self.agents = []
        for _ in range(self.num_agents):
            agent = {
                'pos': np.array([self.grid_size // 2, self.grid_size // 2]),
                'alive': True,
                'homeo': {
                    'hunger': {'value': 0.7, 'setpoint': 0.5, 'decay': 0.003},
                    'thirst': {'value': 0.7, 'setpoint': 0.5, 'decay': 0.004},
                    'energy': {'value': 0.9, 'setpoint': 0.8, 'decay': 0.002}
                }
            }
            self.agents.append(agent)

    def _get_nearest_dist(self, pos: np.ndarray, resources: List[Tuple[int, int]]) -> float:
        if not resources:
            return 1.0
        dists = [np.linalg.norm(pos - np.array(r)) for r in resources]
        max_d = np.sqrt(2 * self.grid_size ** 2)
        return min(dists) / max_d

    def _get_obs(self, agent) -> np.ndarray:
        pos = agent['pos']
        return np.array([
            pos[0] / self.grid_size,
            pos[1] / self.grid_size,
            agent['homeo']['hunger']['value'],
            agent['homeo']['thirst']['value'],
            agent['homeo']['energy']['value'],
            self._get_nearest_dist(pos, self.food_sources),
            self._get_nearest_dist(pos, self.water_sources),
            self._get_nearest_dist(pos, self.rest_areas)
        ], dtype=np.float32)

    def _update_homeostasis(self, agent, action: int) -> None:
        for var in agent['homeo'].values():
            var['value'] -= var['decay']
            var['value'] = np.clip(var['value'], 0.0, 1.0)

        current_pos = tuple(agent['pos'])
        if current_pos in self.food_sources and action == 4:
            agent['homeo']['hunger']['value'] = min(1.0, agent['homeo']['hunger']['value'] + 0.4)
        if current_pos in self.water_sources and action == 4:
            agent['homeo']['thirst']['value'] = min(1.0, agent['homeo']['thirst']['value'] + 0.5)
        if current_pos in self.rest_areas and action == 4:
            agent['homeo']['energy']['value'] = min(1.0, agent['homeo']['energy']['value'] + 0.3)

        if action in [0, 1, 2, 3]:
            agent['homeo']['energy']['value'] -= 0.001

    def _calc_reward(self, agent) -> float:
        dev = 0.0
        for var in agent['homeo'].values():
            dev += abs(var['value'] - var['setpoint'])
            if var['value'] < 0.2:
                dev += 10.0 * (0.2 - var['value'])
        return -dev + 0.1

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self._init_agents()
        self._generate_resources()
        observations = [self._get_obs(agent) for agent in self.agents]
        infos = [self._get_info() for _ in range(self.num_agents)]
        return observations, infos

    def step(self, actions: List[int]):
        coop_events = 0
        conflicts = 0

        obs_list = []
        reward_list = []
        terminated = []
        truncated = []
        infos = []

        for idx, (agent, action) in enumerate(zip(self.agents, actions)):
            if not agent['alive']:
                obs_list.append(self._get_obs(agent))
                reward_list.append(0.0)
                terminated.append(True)
                truncated.append(False)
                infos.append(self._get_info(coop_events, conflicts))
                continue

            # Move
            if action == 0:
                agent['pos'][1] = max(0, agent['pos'][1] - 1)
            elif action == 1:
                agent['pos'][1] = min(self.grid_size - 1, agent['pos'][1] + 1)
            elif action == 2:
                agent['pos'][0] = max(0, agent['pos'][0] - 1)
            elif action == 3:
                agent['pos'][0] = min(self.grid_size - 1, agent['pos'][0] + 1)

            self._update_homeostasis(agent, action)

        # After all agents have moved, check for cooperation/conflict
        positions = {}
        for i, agent in enumerate(self.agents):
            pos = tuple(agent['pos'])
            if pos not in positions:
                positions[pos] = []
            positions[pos].append(i)

        for pos, ids in positions.items():
            if len(ids) > 1:
                if pos in self.food_sources + self.water_sources + self.rest_areas:
                    if self.sharing_enabled:
                        coop_events += 1
                    else:
                        conflicts += 1

        # Compute rewards and termination
        for idx, agent in enumerate(self.agents):
            r = self._calc_reward(agent)
            term = False
            for var in agent['homeo'].values():
                if var['value'] <= 0.0:
                    term = True
                    r -= 20.0
                    agent['alive'] = False
                    break
            obs_list.append(self._get_obs(agent))
            reward_list.append(r)
            terminated.append(term)
            truncated.append(False)
            infos.append(self._get_info(coop_events, conflicts))

        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = [True] * self.num_agents

        return obs_list, reward_list, terminated, truncated, infos

    def _get_info(self, coop: int = 0, conflicts: int = 0) -> Dict:
        return {
            'step': self.current_step,
            'cooperation_events': coop,
            'resource_conflicts': conflicts
        }

    def render(self):
        if self.render_mode != "human":
            return
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        for pos in self.food_sources:
            pygame.draw.circle(self.window, (255, 0, 0),
                               (pos[0] * self.cell_size + self.cell_size // 2,
                                pos[1] * self.cell_size + self.cell_size // 2),
                               self.cell_size // 4)
        for pos in self.water_sources:
            pygame.draw.circle(self.window, (0, 0, 255),
                               (pos[0] * self.cell_size + self.cell_size // 2,
                                pos[1] * self.cell_size + self.cell_size // 2),
                               self.cell_size // 4)
        for pos in self.rest_areas:
            pygame.draw.rect(self.window, (0, 255, 0),
                             (pos[0] * self.cell_size + self.cell_size // 4,
                              pos[1] * self.cell_size + self.cell_size // 4,
                              self.cell_size // 2, self.cell_size // 2))

        colors = [(255, 255, 0), (200, 100, 255), (50, 200, 200), (150, 150, 50)]
        for i, agent in enumerate(self.agents):
            color = colors[i % len(colors)]
            pygame.draw.circle(self.window, color,
                               (agent['pos'][0] * self.cell_size + self.cell_size // 2,
                                agent['pos'][1] * self.cell_size + self.cell_size // 2),
                               self.cell_size // 3)

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
