import gym
import numpy as np
from gym import spaces
import pyglet
from pyglet import gl

class BikeBareboneEnv(gym.Env):
    """
    Description: Environment for a linear bike model.

    Observation (state): roll angle, non-intuitive state element from transfer function and speed (phi, x2, v).
        
    Actions: Steering angle (delta).

    Reward: Not any fixed reward. Can for example use LQR cost or something else. However, a larger penalty is always given if the terminal state is reached.

    Starting State: Preferential, see the reset function.

    Episode Termination: if the magnitude of the roll angle gets larger than 30 degees or the number of steps is larger than 100.

    """

    def __init__(self):
        self.state = None
        self.reward = None
        self.action = None
        self.current_iteration = None

        high = np.array([np.pi/2, np.finfo(np.float32).max, 15]) # Upper limits for the state (has no effect when training, just if you for some reason want to sample random states)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32) # state space (or observation space), 3 elements
        self.action_space = spaces.Box(-np.pi/2, np.pi/2, shape=(1,), dtype=np.float32) # action space, 1 element
        
        # The state space matrices were created with Ts = 0.04 s:
        self.A = np.array([[1.015144907891091, 0.070671622176451], [0.431844962338814, 1.015144907891091]], dtype=np.float32) # is independent of the speed
        self.inv_Ac = np.array([[0, 0.093092967291786], [0.568852500000000, 0]], dtype=np.float32) # Inverse of A. Will be used to calculated B_k later, which depend on the speed
        self.B_c_wo_v = np.array([0.872633942893808, 1.000000000000000], dtype=np.float32) # B_c = B_c_wo_v .* [v; v^2], where c stands for continous case
        
        # Parameters for the cost
        self.Q = np.array([[10, 0], [0, 0]])
        self.R = 1
        self.cost_scale_factor = 1e6 # The ppo algorithm seem to perform bad if the numbers get too small, so use this factor to scale up the cost


    # This function takes a step in the envirment based on an action:
    def step(self, action):
        # Make sure that the action (delta) is in interval [-pi/4, pi/4]:
        action = np.clip(action, -np.pi/2, np.pi/2)[0]
        self.action = action.copy() # For the rendering

        state_wo_v = self.state[0:2] # State elements without v
        v = self.state[2]

        # Calculates the reward based on the LQR cost x_k^T*Q*x_k + u_k^T*R*u_k:
        cost = self.cost_scale_factor * (state_wo_v.transpose() @ self.Q @ state_wo_v + action**2*self.R)
        self.reward = -cost

        # Update the states wihtout v (v is fixed for every sequence in this script, so no updating of v):
        state_wo_v = self.A @ state_wo_v + self.B_k * action 

        self.state = np.array([state_wo_v[0], state_wo_v[1], v], dtype=np.float32)

        # If roll angle larger than 30 degrees, then terminate and give a large punishment (100*large_cost):
        if abs(self.state[0]) > np.deg2rad(30):
            self.done = True
            worst_case_phi = np.pi/180*15
            worst_case_state = np.array([worst_case_phi, 0], dtype=np.float32)
            worst_case_action = np.pi/2
            cost = self.cost_scale_factor * (worst_case_state.transpose() @ self.Q @ worst_case_state + worst_case_action**2*self.R)
            self.reward = -100*cost


        # If 100 steps have been taken, then terminate:
        self.current_iteration += 1
        if self.current_iteration == 100:
            self.done = True
        
        return self.state, self.reward, self.done, {}
        

    # Here the initial conditions are set:
    def reset(self, init_state=None):
        # If an inital state is provided:
        if type(init_state) == np.ndarray:
            self.state = init_state
        else:
            #phi_0 = np.pi/180*np.random.uniform(-5, 5)
            phi_0 = np.deg2rad(np.random.choice([-5, 5]))
            #phi_0 = np.deg2rad(5)
            #v_0 = np.random.uniform(0.5, 10)
            #v_0 = np.random.choice([0.5, 5, 10])
            v_0 = 5
            self.state = np.array([phi_0, 0, v_0], dtype=np.float32)        
        
        # Since B_k (x_k+1 = A*x + B_k*delta_k) depend on the speed, it is calculated now that we know the initial speed (we don't change the speed in this script, so will only be one B_k):
        B_c = self.B_c_wo_v * np.array([self.state[2], self.state[2]**2], dtype=np.float32)
        self.B_k = self.inv_Ac @ (self.A - np.eye(2)) @ B_c

        self.reward = 0
        self.done = False
        self.current_iteration = 0
        return self.state