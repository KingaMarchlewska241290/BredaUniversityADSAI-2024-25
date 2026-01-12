"""
OT-2 Gymnasium Environment Wrapper
This wrapper makes the OT-2 simulation compatible with Stable Baselines 3
"""

import gymnasium as gym
import numpy as np
import sys
import os

# Adding parent directory to path to import sim_class:
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim_class import Simulation


class OT2GymWrapper(gym.Env):
    """
    Gymnasium wrapper for OT-2 robot simulation.
    
    The agent learns to move the pipette tip to a target position.
    
    Observation Space:
        - Current position (x, y, z): 3 values
        - Target position (x, y, z): 3 values  
        - Distance to target: 1 value
        Total: 7 values
    
    Action Space:
        - Velocity commands (vx, vy, vz): 3 continuous values in [-0.1, 0.1] m/s
    
    Reward Function:
        - Distance-based reward (exponential, peaks at goal)
        - Progress reward (getting closer)
        - Time penalty (efficiency)
        - Success bonus (reaching goal within 1mm)
        - Stability bonus (staying at goal)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 target_tolerance=0.001,  # 1mm
                 max_steps=500,
                 render_mode=None):
        """
        Initializing the OT-2 Gym environment.
        
        Args:
            target_tolerance: Distance threshold for success (meters)
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ('human' for GUI, None for no GUI)
        """
        super().__init__()
        
        # Initializing the simulation
        # render=True for GUI (visualization), render=False for training (faster):
        self.sim = Simulation(num_agents=1, render=(render_mode == 'human'))
        self.render_mode = render_mode
        self.sim_initialized = True
        
        # Environment parameters:
        self.target_tolerance = target_tolerance
        self.max_steps = max_steps
        
        # OT-2 workspace limits (from documentation)
        # Making these more conservative to avoid boundaries:
        self.workspace_limits = {
            'x': (0.05, 0.25),  # Adding safety margins
            'y': (0.05, 0.25),  # Adding safety margins
            'z': (0.05, 0.18),  # Adding safety margins
        }
        
        # Defining observation space:
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([0.3, 0.3, 0.2, 0.3, 0.3, 0.2, 0.5]),
            dtype=np.float32
        )
        
        # Defining action space:
        self.action_space = gym.spaces.Box(
            low=np.array([-0.1, -0.1, -0.1]),
            high=np.array([0.1, 0.1, 0.1]),
            dtype=np.float32
        )
        
        # Episode tracking:
        self.current_step = 0
        self.current_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.previous_distance = 0.0
        self.consecutive_success = 0
        
        # Performance tracking:
        self.episode_rewards = []
        self.episode_distances = []

    
    def reset(self, seed=None, options=None):
        """
        Reseting the environment to initial state.
        
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Reseting simulation by reconnecting (if it has a reset method, using that instead)
        # For now, I'll just reset tracking variables and let the simulation continue
        
        # Reseting episode tracking:
        self.current_step = 0
        self.consecutive_success = 0
        self.episode_rewards = []
        self.episode_distances = []
        
        # Getting initial position from simulation:
        state = self.sim.run([[0.0, 0.0, 0.0, 0]])
        self.current_pos = self._extract_position(state)
        
        # Generating random target position within workspace:
        self.target_pos = np.array([
            np.random.uniform(*self.workspace_limits['x']),
            np.random.uniform(*self.workspace_limits['y']),
            np.random.uniform(*self.workspace_limits['z'])
        ])
        
        # Calculating initial distance:
        self.previous_distance = np.linalg.norm(self.current_pos - self.target_pos)
        
        # Creating observation:
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def _extract_position(self, state):
        """
        Extracting pipette position from simulation state.
        
        Args:
            state: State dictionary from simulation
            
        Returns:
            position: numpy array [x, y, z]
        """
        # The state structure is: {'robotId_1': {'pipette_position': [x, y, z], ...}}
        try:
            # Finding the robot key (could be 'robotId_1', 'robotId_2', etc.):
            robot_keys = [k for k in state.keys() if k.startswith('robotId_')]
            
            if not robot_keys:
                print(f"Warning: No robotId found in state: {state.keys()}")
                return np.zeros(3)
            
            # Using the first robot (I only have one agent):
            robot_key = robot_keys[0]
            position = np.array(state[robot_key]['pipette_position'])
            return position
            
        except (KeyError, IndexError, TypeError) as e:
            print(f"Warning: Could not extract position from state: {e}")
            print(f"State structure: {state}")
            return np.zeros(3)
    
    def _get_observation(self):
        """
        Constructing observation vector.
        
        Returns:
            observation: numpy array of shape (7,)
        """
        distance = np.linalg.norm(self.current_pos - self.target_pos)
        
        observation = np.concatenate([
            self.current_pos,      # [0:3] current position
            self.target_pos,       # [3:6] target position
            [distance]             # [6] distance to target
        ]).astype(np.float32)
        
        return observation
    
    def _get_info(self):
        """
        Getting additional information dictionary.
        
        Returns:
            info: Dictionary with episode information
        """
        distance = np.linalg.norm(self.current_pos - self.target_pos)
        
        return {
            'distance': distance,
            'current_pos': self.current_pos.copy(),
            'target_pos': self.target_pos.copy(),
            'step': self.current_step,
            'is_success': distance < self.target_tolerance
        }
    
    def step(self, action):
        """
        Executing one step in the environment.
        
        Args:
            action: numpy array [vx, vy, vz] - velocity commands
            
        Returns:
            observation: New observation after action
            reward: Reward for this step
            terminated: Whether episode ended (success or failure)
            truncated: Whether episode was cut off (max steps)
            info: Additional information
        """
        self.current_step += 1
        
        # Clipping action to valid range (safety):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Scaling down actions for more conservative movement:
        action = action * 0.5  # Reduces velocity by 50%
        
        # Converting action to simulation command format:
        # Format: [velocity_x, velocity_y, velocity_z, drop_command]
        sim_action = [action[0], action[1], action[2], 0]  # drop_command = 0
        
        # Executing action in simulation:
        state = self.sim.run([sim_action], num_steps=1)
        
        # Updating current position:
        self.current_pos = self._extract_position(state)
        
        # Calculating distance to target:
        distance = np.linalg.norm(self.current_pos - self.target_pos)

        # Calculating reward:
        reward = self._calculate_reward(distance)
        
        # Tracking distance history:
        self.episode_distances.append(distance)
        self.episode_rewards.append(reward)

        # Checking if agent is at goal:
        if distance < self.target_tolerance:
            self.consecutive_success += 1
        else:
            self.consecutive_success = 0
        
        # Checking termination conditions:
        terminated = False
        truncated = False
        
        # Success: reached goal and stayed there:
        if distance < self.target_tolerance and self.consecutive_success >= 10:
            terminated = True
            reward += 100  # Big success bonus
        
        # Out of bounds - give penalty but DON'T terminate (allow exploration):
        if self._is_out_of_bounds(self.current_pos):
            reward -= 5  # Small penalty, but keep episode running
        
        # Max steps reached:
        elif self.current_step >= self.max_steps:
            truncated = True
        
        # Updating previous distance for next step:
        self.previous_distance = distance
        
        # Getting observation and info:
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    

    def _calculate_reward(self, distance):
        """
        Calculating reward based on current state.
        
        Multi-component reward function:
        1. Distance reward: Exponential reward that peaks at goal
        2. Progress reward: Reward for getting closer
        3. Time penalty: Small penalty per step for efficiency
        4. Success bonus: Large reward for being within tolerance
        5. Stability bonus: Reward for staying at goal
        
        Args:
            distance: Current distance to target
            
        Returns:
            reward: Scalar reward value
        """
        # Component 1: Distance-based reward (exponential):
        # This creates a strong gradient as it gets closer to the goal:
        distance_reward = np.exp(-10 * distance)
        
        # Component 2: Progress reward:
        # Positive if it got closer, negative if it moved away:
        progress = self.previous_distance - distance
        progress_reward = progress * 10  # Scaling up the progress signal
        
        # Component 3: Time penalty (encourage efficiency):
        time_penalty = -0.01
        
        # Component 4: Success bonus (within 1mm):
        success_bonus = 0
        if distance < self.target_tolerance:
            success_bonus = 10  # Good reward for reaching goal
        
        # Component 5: Stability bonus (staying at goal):
        stability_bonus = 0
        if distance < self.target_tolerance and self.consecutive_success > 5:
            stability_bonus = 5
        
        # Total reward (weighted sum):
        total_reward = (
            10.0 * distance_reward +    # Primary guidance signal
            5.0 * progress_reward +      # Encourage improvement
            time_penalty +               # Encourage speed
            success_bonus +              # Reward reaching goal
            stability_bonus              # Reward stability
        )
        
        return total_reward
    
    def _is_out_of_bounds(self, position):
        """
        Checking if position is outside workspace limits.
        
        Args:
            position: numpy array [x, y, z]
            
        Returns:
            out_of_bounds: Boolean
        """
        x, y, z = position
        
        x_out = x < self.workspace_limits['x'][0] or x > self.workspace_limits['x'][1]
        y_out = y < self.workspace_limits['y'][0] or y > self.workspace_limits['y'][1]
        z_out = z < self.workspace_limits['z'][0] or z > self.workspace_limits['z'][1]
        
        return x_out or y_out or z_out
    
    def render(self):
        """
        Rendering the environment (PyBullet handles this automatically).
        """
        # PyBullet simulation already provides visualization
        pass
    
    def close(self):
        """
        Cleaning up the environment.
        """
        # Disconnecting PyBullet if needed
        pass