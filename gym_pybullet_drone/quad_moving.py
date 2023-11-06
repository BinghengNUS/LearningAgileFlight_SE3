## this file helps quadrotor traverse through a moving narrow window
## including kalman filter, coodinate transformation, get the solution

import numpy as np
from solid_geometry import *
from quad_model import*

class kalman:
    def __init__(self,velo_devia):
        A = np.array([[1,0.1],[0,1]])
        E = np.array([0,0.1])
        C = np.array([1,0])
        sw = velo_devia
        R1 = np.matmul(sw*E.T,sw*E)

        Kf = np.zeros(2,60)
        K = np.zeros(2,60)
        P = np.zeros(2,2,60)
        Pm = np.zeros(2,2,60)

        Pm[:,:,0] = np.array([[1e5,0],[0,1e5]])
        for i in range(60):
            Kf[:,i] = np.matmul(np.matmul(Pm[:,:,i],C.T),1)
        self. X = velo_devia

    def v_es(self, position):
        return position

def solver(model, quad_state, final_point, gate1, velo, w ):
    velo = np.array(velo)

    t_guess = magni(gate1.centroid-quad_state[0:3])/3
    
    t1 = t_guess
    gate_x = gate(gate1.translate_out(velo*t1))
    gate_x.rotate_y(w*t1)

    inputs = np.zeros(18)
    inputs[16] = magni(gate_x.gate_point[0,:]-gate_x.gate_point[1,:])
    inputs[17] = atan((gate_x.gate_point[0,2]-gate_x.gate_point[1,2])/(gate_x.gate_point[0,0]-gate_x.gate_point[1,0]))
    inputs[0:13] = gate_x.transform(quad_state)
    inputs[13:16] = gate_x.t_final(final_point)
    t2 = model(inputs).data.numpy()[6]

    while abs(t2-t1)>0.01:
        t1 += (t2-t1)/2
        gate_x = gate(gate1.translate_out(velo*t1))
        gate_x.rotate_y(w*t1)

        inputs = np.zeros(18)
        inputs[16] = magni(gate_x.gate_point[0,:]-gate_x.gate_point[1,:])
        inputs[17] = atan((gate_x.gate_point[0,2]-gate_x.gate_point[1,2])/(gate_x.gate_point[0,0]-gate_x.gate_point[1,0]))
        inputs[0:13] = gate_x.transform(quad_state)
        inputs[13:16] = gate_x.t_final(final_point)
        t2 = model(inputs).data.numpy()[6]
    
    return t1
