##this file is the package about neural network

from cmath import tan
from math import cos, pi, sin, sqrt, tan
import math
from numpy import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from quad_policy import Rd2Rp
from quad_model import toQuaternion
from solid_geometry import norm,magni
from solid_geometry import plane
from scipy.spatial.transform import Rotation as R

## sample an input for the neural network 1
def nn_sample(init_pos=None,final_pos=None,init_angle=None):
    inputs = np.zeros(9)
    if init_pos is None:
        inputs[0:3] = np.random.uniform(-5,5,size=3) + np.array([0,-9,0]) #-5~5, -9
    else:
        inputs[0:3] = init_pos
    ## random final postion 
    if final_pos is None:
        inputs[3:6] = np.random.uniform(-2,2,size=3) + np.array([0,6,0]) #-2~2, 6
    else:
        inputs[3:6] = final_pos
    ## random initial yaw angle of the quadrotor
    inputs[6] = np.random.uniform(-0.1,0.1)
    ## random width of the gate
    inputs[7] = np.clip(np.random.normal(0.9,0.3),0.5,1.25)
    # inputs[7] = np.random.uniform(0.7,1.2)
    ## random pitch angle of the gate
    angle = np.clip(1.3*(1.2-inputs[7]),0,pi/3)
    angle1 = (pi/2-angle)/3
    judge = np.random.normal(0,1)
    if init_angle is None:
        if judge > 0:
            inputs[8] = np.clip(np.random.normal(angle + angle1, 2*angle1/3),angle,pi/2)
            # inputs[8] = np.random.uniform(angle - angle1, angle + angle1)
        else:
            inputs[8] = np.clip(np.random.normal(-angle - angle1, 2*angle1/3),-pi/2,-angle)
    else:
        inputs[8] = init_angle
    
    # inputs[8] = 0.8879
    return inputs

## define the expected output of an input (for pretraining)
def t_output(inputs):
    inputs = np.array(inputs)
    outputs = np.zeros(7)
    #outputs[5] = math.tan(inputs[6]/2)
    ## traversal time is propotional to the distance of the centroids
    outputs[6] = np.clip(round(magni(inputs[0:3])/4,1),2,4)
    return outputs

## sample a random gate (not necessary in our method) (not important)
def gene_gate():
    point1 = np.array([0,0,0])
    #generate diagonal line and point3
    dia_line = np.random.uniform(1.5,3)
    point3 = np.array([dia_line,0,0])
    # generate point2
    point2x = np.random.normal(dia_line/2,dia_line/2)
    point2z = np.random.uniform(0,dia_line)
    point2 = np.array([point2x,0,point2z])
    # generate point4
    point4x = np.random.normal(dia_line/2,dia_line/2)
    point4z = np.random.uniform(-dia_line, 0)
    point4 = np.array([point4x,0,point4z])
    return np.array([point1,point2,point3,point4])


## sample any initial state, final point and 12 elements window (not necessary in our method) (not important)
def con_sample():
    inputs = np.zeros(25)
    # generate first three inouts
    scaling = np.random.uniform(3,16)
    phi = np.random.uniform(0,2*pi)
    theta = np.clip(np.random.normal(pi/2,pi/8,size=1), pi/4, 3*pi/4)
    #transformation
    inputs[0] = scaling*sin(theta)*cos(phi)
    inputs[1] = scaling*sin(theta)*sin(phi)
    inputs[2] = scaling*cos(theta)
    beta = np.random.uniform(0,2*pi)
    rotation1 = np.array([[cos(beta),0,sin(beta)],[0,1,0],[-sin(beta),0,cos(beta)]])
    rotation2 = np.array([[cos(phi-pi/2),-sin(phi-pi/2),0],[sin(phi-pi/2),cos(phi-pi/2),0],[0,0,1]])
    rotation  = np.matmul(rotation2,rotation1)
    # generate rotation pair
    l = norm(np.random.normal(0,1,size=3))
    a = np.random.normal(0,pi/16)
    r = R.from_rotvec(a * l)
    rotation = np.matmul(r.as_matrix(),rotation)
    # generate translation
    length = np.random.uniform(2,scaling-1) 
    tranlation1 = np.array([length*sin(theta)*cos(phi),length*sin(theta)*sin(phi),length*cos(theta)])
    tranlation = tranlation1 + np.random.normal(0,1,size=3)
    # generate real obstacle
    gate = gene_gate()
    for i in range(4):
        gate[i] = np.matmul(rotation,gate[i]) + tranlation
    inputs[3:15] = gate.reshape(12)
        #generate velocity
    inputs[15:18] = np.random.normal(0,3,size=3)
    #generate quaternions
    Rd = np.random.normal(0,0.5,size=3)
    rp = Rd2Rp(Rd)
    inputs[18:22] = toQuaternion(rp[0],rp[1])
    distance = np.random.uniform(0,scaling)
    inputs[22] = distance*sin(theta)*cos(phi)+np.random.normal(0,1)
    inputs[23] = distance*sin(theta)*sin(phi)+np.random.normal(0,1)
    inputs[24] = distance*cos(theta)+np.random.normal(0,1)
    return inputs


## define the class of neural network (2 hidden layers, unit = ReLU)
class network(nn.Module):
    def __init__(self, D_in, D_h1, D_h2, D_out):
        super(network, self).__init__()        
        # D_in : dimension of input layer
        # D_h  : dimension of hidden layer
        # D_out: dimension of output layer
        self.l1 = nn.Linear(D_in, D_h1)
        self.F1 = nn.ReLU()
        self.l2 = nn.Linear(D_h1, D_h2)
        self.F2 = nn.ReLU()
        self.l3 = nn.Linear(D_h2, D_out)

    def forward(self, input):
        # convert state s to tensor
        S = torch.tensor(input, dtype=torch.float) # column 2D tensor
        out = self.l1(S.t()) # linear function requires the input to be a row tensor
        out = self.F1(out)
        out = self.l2(out)
        out = self.F2(out)
        out = self.l3(out)
        return out

    def myloss(self, para, dp):
        # convert np.array to tensor
        Dp = torch.tensor(dp, dtype=torch.float) # row 2D tensor
        loss_nn = torch.matmul(Dp, para)
        return loss_nn




