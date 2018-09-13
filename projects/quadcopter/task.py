import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions
                       and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z)
                             dimensions
            init_angle_velocities: initial radians/second for each of the three
                                   Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, 
                              init_angle_velocities, runtime)
        self.runtime = runtime
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Initial pose
        if init_pose is not None:
            self.init_pos = init_pose
        else:
            self.init_pos = np.array([0., 0., 1.])
        
        # Goal
        if target_pos is not None:
            self.target_pos = target_pos
        else:
            self.target_pos = np.array([0., 0., 10.]) 

    def get_reward(self, termination, reached):
        """Uses current pose of sim to return reward."""
        goal = self.target_pos[2]
        current_z = self.sim.pose[2]
        init_z = self.init_pos[2]
        pct_progress = (current_z - init_z)/(goal - init_z)
        # Exponential reward for altitude gain
        reward = (1 + pct_progress)**2
        # Penalty for floating down
        if self.sim.v[2] < 0:
            reward -= 1
        # Penalty for crashing or timing out that decreases with progress
        if termination:
            reward -= 5.*(1 - pct_progress)
        # Bonus for reaching and exceeding target quickly 
        if reached:
            reward += 5. + (current_z-goal) + 2*(self.runtime-self.sim.time)
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            # Check if out of bounds or out of time
            termination = self.sim.next_timestep(rotor_speeds)
            # Check if goal has been reached
            reached = self.sim.pose[2] > self.target_pos[2]
            # Stop episode if terminated or reached goal
            done = (termination or reached)
            # Update reward
            reward += self.get_reward(termination, reached) 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done, reached

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state