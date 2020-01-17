''' Simple script to test if the BikeLQR_4states environment seems to work as intended.'''

import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.
    
    x[k+1] = A x[k] + B u[k]
    
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151
    
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
    
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    
    return K, X, eigVals


A = np.array([[1.015144907891091, 0.070671622176451], [0.431844962338814, 1.015144907891091]], dtype=np.float32)
B_c_wo_v = np.array([0.872633942893808, 1.000000000000000], dtype=np.float32)
inv_Ac = np.array([[0, 0.093092967291786], [0.568852500000000, 0]], dtype=np.float32)

from BikeLQR_4states import BikeLQR_4statesEnv # import the custom environment class
env = BikeLQR_4statesEnv()
state = env.reset()

Ts = 0.04
nr_time_steps = 100
cumulative_reward = 0
phi_list = [state[0]*180/np.pi]
delta_list = [np.rad2deg(state[3])]
for i in range(nr_time_steps):
    v = state[2]
    B_c = B_c_wo_v * np.array([v, v**2], dtype=np.float32)
    B_k = inv_Ac @ (A - np.eye(2)) @ B_c
    B_k = B_k.reshape((2,1))
    Q = np.array([[10, 0], [0, 0]])
    R = 1
    K, X, eigVals = dlqr(A,B_k,Q,R)
    K = np.squeeze(np.asarray(K))

    action = -K@state[0:2]
    state, reward, done, _ = env.step([action])

    print('reward:', reward)
    print('phi:', state[0].item()*180/np.pi)

    phi_list.append(state[0].item()*180/np.pi)
    delta_list.append(np.rad2deg(state[3]))
    
    cumulative_reward += reward
    print('Cumulative reward', cumulative_reward)
    
    if done:
        break


t = np.arange(0, len(phi_list)*Ts, Ts)
fig, axes = plt.subplots(1,3, figsize=(14,8))
fig.suptitle('v = ' + str(v), fontsize=16)

axes[0].plot(t, phi_list)
axes[0].set_title(r'$\varphi$', fontsize=14)

axes[1].plot(t, delta_list)
axes[1].set_title('$\delta$', fontsize=14)

axes[2]. plot(t[1:], np.diff(delta_list)/Ts)
axes[2].set_title('$\dot{\delta}$', fontsize=14)

plt.show()