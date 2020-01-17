import scipy.linalg
import numpy as np
import pandas as pd

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

def get_K(v, A, B_c_wo_v, inv_Ac, Q, R):

    B_c = B_c_wo_v * np.array([v, v**2], dtype=np.float32)
    B_k = inv_Ac @ (A - np.eye(2)) @ B_c
    K, X, eigVals = dlqr(A, B_k.reshape((2,1)), Q, R)
    K = np.squeeze(np.asarray(K))
    return K

def get_optimal_sequence(init_state, env, Q, R, nr_steps_eval):

    state = env.reset(init_state=init_state)
    v = state[2]
    A = env.A
    B_c_wo_v = env.B_c_wo_v
    inv_Ac = env.inv_Ac

    done = False
    phi_list = [state[0]]
    delta_list = [state[3]]
    for _ in range(nr_steps_eval):
        
        v = state[2]
        K = get_K(v, A, B_c_wo_v, inv_Ac, Q, R)
        
        action = -K@state[0:2]
        state, reward, done, _ = env.step([action])
        phi_list.append(state[0].item())
        delta_list.append(state[3])

        if done:
            break
    
    return phi_list, delta_list
