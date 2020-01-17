import gym
import numpy as np
from gym import spaces
import pyglet
from pyglet import gl

class Bike_4statesEnv(gym.Env):
    """
    Description: Environment for a bike where saturation can be applied on the steering angle rate.

    Observation (state): roll angle, non-intuitive state element from transfer function, speed and current steering angle (phi, x2, v and delta_old).
        
    Actions: Steering angle (delta).

    Reward: Not any fixed reward. Can for example use LQR cost or something else. However, a larger penalty is always given if the terminal state is reached.

    Starting State: Preferential, see the reset function.

    Episode Termination: if the magnitude of the roll angle gets larger than 30 degees.

    """

    def __init__(self):
        super(Bike_4statesEnv, self).__init__()
        self.state = None
        self.reward = None
        self.viewer = None
        self.action = None
        self.current_iteration = None

        self.length = 0.5 # Legnth of bike (only use for the rendering)
        
        self.dt = 0.04 # Sample time (seconds)
        self.current_iteration = 0

        # Scenario parameters
        self.top_rate = np.deg2rad(100) # Maximum allowed steering rate (rad/s)

        # Cost parameters
        self.Q = np.array([[1, 0], [0, 0]])
        self.R = 1
        self.cost_scale_factor = 1e6 # The results can severely be affected if you don't have a large enough scale factor for the cost. So the PPO-algorithm doesn't seem to handle
                                     # small values well
        
        # Worst case scenario parameters for when the termial state is reached
        worst_case_phi = np.deg2rad(30)
        worst_case_state = np.array([worst_case_phi, 0], dtype=np.float32)
        worst_case_action = np.pi/2 # Change
        self.worst_case_cost = self.cost_scale_factor * (worst_case_state.transpose() @ self.Q @ worst_case_state + worst_case_action**2*self.R)

        high = np.array([np.pi/2, np.finfo(np.float32).max, 15, np.pi/2])
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(-np.pi/2, np.pi/2, shape=(1,), dtype=np.float32)
        
        # The state space matrices were created with Ts = 0.04 s
        self.A = np.array([[1.015144907891091, 0.070671622176451], [0.431844962338814, 1.015144907891091]], dtype=np.float32) # is independent of the speed
        self.inv_Ac = np.array([[0, 0.093092967291786], [0.568852500000000, 0]], dtype=np.float32) # Inverse of A. Will be used to calculated B_k later, which depend on the speed
        self.B_c_wo_v = np.array([0.872633942893808, 1.000000000000000], dtype=np.float32) # B_c = B_c_wo_v .* [v; v^2], where c stands for continous case


    # This function takes a step in the envirment based on an action:
    def step(self, action):
    
        # Extract the old action from the state:
        old_action = self.state[3]

        # Convert the action to a scalar:
        action = action[0]

        # Restricts the steering rate so it is less than the top_rate. The steering rate is approximated using euler backwards:
        if abs(action - old_action) > self.top_rate*self.dt:
            action = old_action + np.sign(action - old_action)*self.top_rate*self.dt

        # Make sure that the action (delta) is in interval [-pi/4, pi/4]:
        action = np.clip(action, -np.pi/4, np.pi/4)
        self.action = action.copy() # Only for the rendering    

        # Replace old_action to the new for the state:
        old_action = action.copy()

        # Extract x and v from the state
        state_wo_v = self.state[0:2]
        v = self.state[2]

        # Calculate the cost and reward 
        cost = self.cost_scale_factor * (state_wo_v.transpose() @ self.Q @ state_wo_v + action**2*self.R) #+ LQR cost
        self.reward = -cost

        # Calculate the new state
        B_c = self.B_c_wo_v * np.array([v, v**2], dtype=np.float32)
        B_k = self.inv_Ac @ (self.A - np.eye(2)) @ B_c
        state_wo_v = self.A @ state_wo_v + B_k * action # action is scalar
            
        self.state = np.array([state_wo_v[0], state_wo_v[1], v, old_action], dtype=np.float32)

        # If roll angle larger than 30 degrees, then terminate and severly punish so that it's never beneficial for the agent to terminate:
        if np.abs(self.state[0]) > np.deg2rad(30):
            self.done = True
            self.reward = -100 * self.worst_case_cost
        
        ############# Actually I overwrite the reward based on the LQR cost with this simpler version, since it is faster to train. I still leave the code for the LQR cost though #########
        self.reward = 1 # Get a reward for just staying up (i.e. as long as it's not reaching 30 deg roll angle and terminating, it will be rewarded)
        if np.abs(self.state[0]) < np.deg2rad(0.1): 
            self.reward = 100 # Then it gets even more reward if it can be within small intervals
        elif np.abs(self.state[0]) < np.deg2rad(1): 
            self.reward = 10
        ############ Just comment out the lines inside these border to use the LQR cost ########################

        # If 100 steps have been taken, then terminate:
        self.current_iteration += 1
        if self.current_iteration == 100:
            self.done = True

        return self.state, self.reward, self.done, {}
        
    # Here you specify the initial conditions you want to train the agent on:
    def reset(self, init_state=None):
        # If initial state is specified:
        if type(init_state) == np.ndarray:
            self.state = init_state
        else:
            #phi_0 = np.deg2rad(np.random.uniform(-5, 5))
            phi_0 = np.random.choice([np.deg2rad(-5), np.deg2rad(5)])
            v_0 = 0.9
            #v_0 = np.random.choice([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            #v_0 = np.random.uniform(0.5, 10)
            self.state = np.array([phi_0, 0, v_0, 0], dtype=np.float32)        
        
        self.reward = 0
        self.done = False
        self.current_iteration = 0
        return self.state
        
    # This function is used to visualize the balacing:
    def render(self, mode='human'):
        screen_width  = 600
        screen_height = 400

        axleoffset = 30.0/4.0
        world_width = 2.4*2
        scale = screen_width/world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Text
            #self.score_label = pyglet.text.Label("Nu går det fort!!", font_size=12,
            #    x=0, y=0, anchor_x='center', anchor_y='center',
            #    color=(255,255,255,255))
            #self.score_label.draw()

            # Cykeln
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(500, 100))
            pole.add_attr(self.poletrans)
            self.viewer.add_geom(pole)
            self._pole_geom = pole

            # Styret
            styre1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            styre2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            styre1.set_color(.8,.6,.4)
            styre2.set_color(.8,.6,.4)
            self.styre1trans = rendering.Transform(translation=(200, 100), rotation=3.14/2)
            self.styre2trans = rendering.Transform(translation=(200, 100), rotation=-3.14/2)
            styre1.add_attr(self.styre1trans)
            styre2.add_attr(self.styre2trans)
            self.viewer.add_geom(styre1)
            self.viewer.add_geom(styre2)
            self._styre1_geom = styre1
            self._styre2_geom = styre2
        
        if self.state is None: return None

        pole = self._pole_geom
        styre1 = self._styre1_geom
        styre2 = self._styre2_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]
        styre1.v = [(l,b), (l,t), (r,t), (r,b)]
        styre2.v = [(l,b), (l,t), (r,t), (r,b)]
        
        x = self.state
        self.poletrans.set_rotation(-x[0])

        #self.score_label.text = "Nu går det fort!!"
        #self.score_label.text = "%04i" % self.reward
        #self.score_label.draw()

        if self.action is not None:
            self.styre1trans.set_rotation(self.action+np.pi/2)
            self.styre2trans.set_rotation(self.action-np.pi/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
