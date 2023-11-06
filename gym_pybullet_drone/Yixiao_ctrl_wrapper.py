"""
Wrapper that wraps main process into a class for compact representation and variable reusing.
Inherited from control class 'SimplePIDControl.py'.
Edited by: ZSN

Version: 2.0
"""
import shutil

import pybullet as p
import time

from quad_policy import *
from quad_nn import *
from quad_moving import *

import sys
sys.path.append('../')

from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.enums import DroneModel


class YXCtrlWrapper(SimplePIDControl):
    def __init__(self,
                 drone_model: DroneModel,
                 g: float = 9.8,
                 # required paras
                 gate_paras=None,  # in dict form
                 half_gt_hei=1,
                 gate_v=np.array([0.5, 0.3, 0.4]),
                 gate_w=0.5,
                 relative_ori=np.zeros(3),
                 model_file="nn3_1.pth",
                 ctrl_every_n_steps=1,
                 replicate_sim=False,
                 ):
        super().__init__(drone_model=drone_model, g=g)

        # initialization

        INPUT_PREFERENCES = 'last_inputs.npy'
        BACKUP_FOLDER = 'sim_backup'

        # sample the input
        if replicate_sim and os.path.exists(INPUT_PREFERENCES):
            # backup all sim settings that are featured
            with open(INPUT_PREFERENCES, 'rb') as f:
                is_new_input, inputs = np.load(f)[0], np.load(f)
            # only backup those not been backup before
            if is_new_input:
                with open(INPUT_PREFERENCES, 'wb') as f:
                    np.save(f, np.array([False]))
                    np.save(f, inputs)
                backup_path = list(os.path.splitext(INPUT_PREFERENCES))
                backup_path.insert(1, time.strftime('_%y%m%d_%H%M%S'))
                backup_path = os.path.join(BACKUP_FOLDER, ''.join(backup_path))
                shutil.copy(INPUT_PREFERENCES, backup_path)
        else:
            inputs = nn_sample_pybullet(**gate_paras)
            print(f'>>> New simulation starts, using inputs: {inputs}')
            # note: the input file in 2.0 is incompatible with previous
            with open(INPUT_PREFERENCES, 'wb') as f:
                np.save(f, np.array([True]))  # flag of newly generated settings
                np.save(f, inputs)

        self.start_point = inputs[0:3]
        self.final_point = inputs[3:6]
        # initial obstacle
        gate_point0 = np.array([[-inputs[7]/2, 0, half_gt_hei], [inputs[7]/2, 0, half_gt_hei],
                                [inputs[7]/2, 0, -half_gt_hei], [-inputs[7]/2, 0, -half_gt_hei]])
        gate1 = gate(gate_point0)
        gate1.rotate_y(inputs[8])
        gate_point = gate1.gate_point
        gate1 = gate(gate_point)
        self.UAV_ini_yaw = inputs[6]  # for UAV initialization
        self.gate_width = inputs[7]  # new: two for gate model loading
        self.gate_pitch = inputs[8]

        self.FILE = model_file
        self.model = torch.load(self.FILE)

        # define the kinematics of the narrow window
        self.v = gate_v
        self.w = gate_w
        self.gate_move = gate1.move(v=self.v, w=self.w)

        self.relative_ori = relative_ori  # relative frame origin

        self.l, self.c = 0.35, 0.0245  # UAV parameters used in training

        self.u = [0, 0, 0, 0]

        self.ctrl_every_n_steps = ctrl_every_n_steps

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3),
                       ):
        gate_n = gate(self.gate_move[self.control_counter])

        state = np.hstack((cur_pos - self.relative_ori,
                           cur_vel,
                           cur_quat[[3, 0, 1, 2]],  # Quaternion definition is different from normal
                           angu_vel_tran_w2b(cur_ang_vel, cur_quat),  # w transform from Euler to body
                           ))
        ## solve for the traversal time
        t = solver(self.model, state, self.final_point, gate_n, self.v, self.w)
        # print('step', self.control_counter, 'tranversal time=', t)  # test

        # obtain the future traversal window state
        gate_n.translate(t * self.v)
        gate_n.rotate_y(t * self.w)
        # obtain the state in window frame
        inputs = np.zeros(18)
        inputs[16] = magni(gate_n.gate_point[0, :]-gate_n.gate_point[1, :])
        inputs[17] = atan((gate_n.gate_point[0, 2]-gate_n.gate_point[1, 2]) /
                          (gate_n.gate_point[0, 0]-gate_n.gate_point[1, 0]))
        inputs[0:13] = gate_n.transform(state)
        inputs[13:16] = gate_n.t_final(self.final_point)

        out = self.model(inputs).data.numpy()

        # solve the mpc problem and get the control command
        quad2 = run_quad(goal_pos=inputs[13:16], horizon=50)
        self.u = quad2.get_input(inputs[0:13], self.u, out[0:3], out[3:6], out[6])

        # control adjustment: use 4x1 thrust and torques as input under DynAviary
        u = np.dot(np.diag([1, -self.l/2, self.l/2, -self.c]).dot(self.A), self.u)

        self.control_counter += self.ctrl_every_n_steps

        return u, t


def nn_sample_pybullet(start_p, st_p_range, end_p, end_p_range,
                       gate_wid_rand=None, gate_wid_lim=None):
    """
    Parameterized sampling function for Pybullet simulation (modified from quad_nn.py func nn_sample).
    """
    if gate_wid_rand is None:
        gate_wid_rand = [0.9, 0.2]
    if gate_wid_lim is None:
        gate_wid_lim = [0.8, 1.5]
    inputs = np.zeros(9)
    inputs[0:3] = np.array([3, start_p, -0.2]) + np.random.uniform(-st_p_range, st_p_range, size=3) #np.array([2, 1, 2]) + \
    ## random final postion
    inputs[3:6] = np.random.uniform(-end_p_range, end_p_range, size=3) + \
                  np.array([0, end_p, 0])
    ## random initial yaw angle
    inputs[6] = np.random.uniform(-pi/6, pi/6)
    ## random width of the gate
    inputs[7] = np.clip(np.random.normal(gate_wid_rand[0], gate_wid_rand[1]),
                        gate_wid_lim[0], gate_wid_lim[1])
    ## random pitch angle of the gate
    angle = np.clip(1.3 * (1.2 - inputs[7]), 0, pi / 3)
    angle1 = (pi / 2 - angle) / 3
    judge = np.random.normal(0, 1)
    if judge > 0:
        inputs[8] = np.clip(np.random.normal(angle + angle1, 2 * angle1 / 3), angle, pi / 2)
        # inputs[8] = -30/180*pi
    else:
        inputs[8] = np.clip(np.random.normal(-angle - angle1, 2 * angle1 / 3), -pi / 2, -angle)
        # inputs[8] = -30/180*pi

    return inputs


def angu_vel_tran_w2b(d_rpy, quat):
    # copied and modified from previous projects
    rpy = p.getEulerFromQuaternion(quat)
    Q_inv = np.array([[1,               0,                 -np.sin(rpy[1])],
                      [0,  np.cos(rpy[0]), np.sin(rpy[0]) * np.cos(rpy[1])],
                      [0, -np.sin(rpy[0]), np.cos(rpy[0]) * np.cos(rpy[1])]
                      ])
    omega_b = np.dot(Q_inv, d_rpy)
    return omega_b
