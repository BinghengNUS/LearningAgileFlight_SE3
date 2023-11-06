##this file is to generate model of quadrotor

from casadi import *
import casadi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from scipy.spatial.transform import Rotation as R
from solid_geometry import norm
from math import sqrt

# quadrotor (UAV) environment
class Quadrotor:
    def __init__(self, project_name='my UAV'):
        self.project_name = 'my uav'

        # define the state of the quadrotor
        rx, ry, rz = SX.sym('rx'), SX.sym('ry'), SX.sym('rz')
        self.r_I = vertcat(rx, ry, rz)
        vx, vy, vz = SX.sym('vx'), SX.sym('vy'), SX.sym('vz')
        self.v_I = vertcat(vx, vy, vz)
        # quaternions attitude of B w.r.t. I
        q0, q1, q2, q3 = SX.sym('q0'), SX.sym('q1'), SX.sym('q2'), SX.sym('q3')
        self.q = vertcat(q0, q1, q2, q3)
        wx, wy, wz = SX.sym('wx'), SX.sym('wy'), SX.sym('wz')
        self.w_B = vertcat(wx, wy, wz)
        # define the quadrotor input
        f1, f2, f3, f4 = SX.sym('f1'), SX.sym('f2'), SX.sym('f3'), SX.sym('f4')
        self.T_B = vertcat(f1, f2, f3, f4)

    def initDyn(self, Jx=None, Jy=None, Jz=None, mass=None, l=None, c=None):
        # global parameter
        g = 9.78
        # parameters settings
        parameter = []
        if Jx is None:
            self.Jx = SX.sym('Jx')
            parameter += [self.Jx]
        else:
            self.Jx = Jx

        if Jy is None:
            self.Jy = SX.sym('Jy')
            parameter += [self.Jy]
        else:
            self.Jy = Jy

        if Jz is None:
            self.Jz = SX.sym('Jz')
            parameter += [self.Jz]
        else:
            self.Jz = Jz

        if mass is None:
            self.mass = SX.sym('mass')
            parameter += [self.mass]
        else:
            self.mass = mass

        if l is None:
            self.l = SX.sym('l')
            parameter += [self.l]
        else:
            self.l = l

        if c is None:
            self.c = SX.sym('c')
            parameter += [self.c]
        else:
            self.c = c

        self.dyn_auxvar = vcat(parameter)

        # Angular moment of inertia
        self.J_B = diag(vertcat(self.Jx, self.Jy, self.Jz))
        # Gravity
        self.g_I = vertcat(0, 0, -g)
        # Mass of rocket, assume is little changed during the landing process
        self.m = self.mass

        # total thrust in body frame
        thrust = self.T_B[0] + self.T_B[1] + self.T_B[2] + self.T_B[3]
        self.thrust_B = vertcat(0, 0, thrust)
        # total moment M in body frame
        Mx = -self.T_B[1] * self.l / 2 + self.T_B[3] * self.l / 2
        My = -self.T_B[0] * self.l / 2 + self.T_B[2] * self.l / 2
        Mz = (self.T_B[0] - self.T_B[1] + self.T_B[2] - self.T_B[3]) * self.c
        self.M_B = vertcat(Mx, My, Mz)

        #Mx = self.T_B[0] * sqrt(2)*self.l / 4 -self.T_B[1] * sqrt(2)*self.l / 4 - self.T_B[2] * sqrt(2)*self.l / 4 + self.T_B[3] * sqrt(2)*self.l / 4
        #My = self.T_B[0] * sqrt(2)*self.l / 4 +self.T_B[1] * sqrt(2)*self.l / 4 - self.T_B[2] * sqrt(2)*self.l / 4 - self.T_B[3] * sqrt(2)*self.l / 4
        #Mz = (self.T_B[0] + self.T_B[1] - self.T_B[2] - self.T_B[3]) * self.c


        # cosine directional matrix
        C_B_I = self.dir_cosine(self.q)  # inertial to body
        C_I_B = transpose(C_B_I)  # body to inertial

        # Newton's law
        dr_I = self.v_I
        dv_I = 1 / self.m * mtimes(C_I_B, self.thrust_B) + self.g_I
        # Euler's law
        dq = 1 / 2 * mtimes(self.omega(self.w_B), self.q)
        dw = mtimes(inv(self.J_B), self.M_B - mtimes(mtimes(self.skew(self.w_B), self.J_B), self.w_B))

        self.X = vertcat(self.r_I, self.v_I, self.q, self.w_B)
        self.U = self.T_B
        self.f = vertcat(dr_I, dv_I, dq, dw)

    def initCost(self, wrt=None, wqt=None, wrf=None, wvf=None, wqf=None, wwf=None, \
        wthrust=0.5,goal_pos=[0,9,5],goal_velo = [0,0,0],goal_atti=[0,[1,0,0]]):
        #traverse
        parameter = []
        if wrt is None:
            self.wrt = SX.sym('wrt')
            parameter += [self.wr]
        else:
            self.wrt = wrt

        if wqt is None:
            self.wqt = SX.sym('wqt')
            parameter += [self.wq]
        else:
            self.wqt = wqt

        #path
        if wrf is None:
            self.wrf = SX.sym('wrf')
            parameter += [self.wrf]
        else:
            self.wrf = wrf
        
        if wvf is None:
            self.wvf = SX.sym('wvf')
            parameter += [self.wvf]
        else:
            self.wvf = wvf
        
        if wqf is None:
            self.wqf = SX.sym('wqf')
            parameter += [self.wqf]
        else:
            self.wqf = wqf
        
        if wwf is None:
            self.wwf = SX.sym('wwf')
            parameter += [self.wwf]
        else:
            self.wwf = wwf

        self.cost_auxvar = vcat(parameter)

        ## goal cost
        # goal position in the world frame
        self.goal_r_I = goal_pos
        self.cost_r_I_g = dot(self.r_I - self.goal_r_I, self.r_I - self.goal_r_I)

        # goal velocity
        goal_velo = goal_velo
        self.goal_v_I = goal_velo
        self.cost_v_I_g = dot(self.v_I - self.goal_v_I, self.v_I - self.goal_v_I)

        # final attitude error
        self.goal_q = toQuaternion(goal_atti[0],goal_atti[1])
        goal_R_B_I = self.dir_cosine(self.goal_q)
        R_B_I = self.dir_cosine(self.q)
        self.cost_q_g = trace(np.identity(3) - mtimes(transpose(goal_R_B_I), R_B_I))

        # auglar velocity cost
        self.goal_w_B = [0, 0, 0]
        self.cost_w_B_g = dot(self.w_B - self.goal_w_B, self.w_B - self.goal_w_B)


        # the thrust cost
        self.cost_thrust = dot(self.T_B, self.T_B)

        self.thrust_cost = wthrust * self.cost_thrust

        ## the final (goal) cost
        self.goal_cost = self.wrf * self.cost_r_I_g + \
                         self.wvf * self.cost_v_I_g + \
                         self.wwf * self.cost_w_B_g + \
                         self.wqf * self.cost_q_g 
        self.final_cost = self.wrf * self.cost_r_I_g + \
                          self.wvf * self.cost_v_I_g + \
                          self.wwf * self.cost_w_B_g + \
                          self.wqf * self.cost_q_g

    def init_TraCost(self, tra_pos = [0, 0, 5], tra_atti = [0.7,[0,1,0]]):
        ## traverse cost
        # traverse position in the world frame
        self.tra_r_I = tra_pos[0:3]
        self.cost_r_I_t = dot(self.r_I - self.tra_r_I, self.r_I - self.tra_r_I)

        # traverse attitude error
        self.tra_q = toQuaternion(tra_atti[0],tra_atti[1])
        tra_R_B_I = self.dir_cosine(self.tra_q)
        R_B_I = self.dir_cosine(self.q)
        self.cost_q_t = trace(np.identity(3) - mtimes(transpose(tra_R_B_I), R_B_I))

        self.tra_cost =   self.wrt * self.cost_r_I_t + \
                            self.wqt * self.cost_q_t

    def setDyn(self, dt):       

        self.dyn = casadi.Function('f',[self.X, self.U],[self.f])
        self.dyn = self.X + dt * self.f
        self.dyn_fn = casadi.Function('dynamics', [self.X, self.U], [self.dyn])

        #M = 4
        #DT = dt/4
        #X0 = casadi.SX.sym("X", self.X.numel())
        #U = casadi.SX.sym("U", self.U.numel())
        # #
        #X = X0
        #for _ in range(M):
            # --------- RK4------------
        #    k1 =DT*self.dyn(X, U)
        #    k2 =DT*self.dyn(X+0.5*k1, U)
        #    k3 =DT*self.dyn(X+0.5*k2, U)
        #    k4 =DT*self.dyn(X+k3, U)
            #
        #    X = X + (k1 + 2*k2 + 2*k3 + k4)/6        
        # Fold
        #self.dyn_fn = casadi.Function('dyn', [X0, U], [X])

    ## below is for animation (demo)
    def get_quadrotor_position(self, wing_len, state_traj):

        # thrust_position in body frame
        r1 = vertcat(wing_len*0.5/ sqrt(2) , wing_len*0.5/ sqrt(2) , 0)
        r2 = vertcat(-wing_len*0.5 / sqrt(2), wing_len*0.5 / sqrt(2), 0)
        r3 = vertcat(-wing_len*0.5 / sqrt(2), -wing_len*0.5 / sqrt(2), 0)
        r4 = vertcat(wing_len*0.5 / sqrt(2), -wing_len*0.5 / sqrt(2), 0)

        # horizon
        horizon = np.size(state_traj, 0)
        position = np.zeros((horizon, 15))
        for t in range(horizon):
            # position of COM
            rc = state_traj[t, 0:3]
            # altitude of quaternion
            q = state_traj[t, 6:10]

            # direction cosine matrix from body to inertial
            CIB = np.transpose(self.dir_cosine(q).full())

            # position of each rotor in inertial frame
            r1_pos = rc + mtimes(CIB, r1).full().flatten()
            r2_pos = rc + mtimes(CIB, r2).full().flatten()
            r3_pos = rc + mtimes(CIB, r3).full().flatten()
            r4_pos = rc + mtimes(CIB, r4).full().flatten()

            # store
            position[t, 0:3] = rc
            position[t, 3:6] = r1_pos
            position[t, 6:9] = r2_pos
            position[t, 9:12] = r3_pos
            position[t, 12:15] = r4_pos

        return position
    
    def get_final_position(self,wing_len, p= None,q = None):
        p = self.tra_r_I
        q = self.tra_q
        r1 = vertcat(wing_len*0.5 / sqrt(2), wing_len*0.5 / sqrt(2), 0)
        r2 = vertcat(-wing_len*0.5 / sqrt(2), wing_len*0.5 / sqrt(2), 0)
        r3 = vertcat(-wing_len*0.5 / sqrt(2), -wing_len*0.5 / sqrt(2), 0)
        r4 = vertcat(wing_len*0.5 / sqrt(2), -wing_len*0.5 / sqrt(2), 0)

        CIB = np.transpose(self.dir_cosine(q).full())
 
        r1_pos = p + mtimes(CIB, r1).full().flatten()   
        r2_pos = p + mtimes(CIB, r2).full().flatten()
        r3_pos = p + mtimes(CIB, r3).full().flatten()
        r4_pos = p + mtimes(CIB, r4).full().flatten()

        position = np.zeros(15)
        position[0:3] = p
        position[3:6] = r1_pos
        position[6:9] = r2_pos
        position[9:12] = r3_pos
        position[12:15] = r4_pos

        return position



    def play_animation(self, wing_len, state_traj, gate_traj1=None, gate_traj2=None,state_traj_ref=None, dt=0.01, \
            point1 = None,point2 = None,point3 = None,point4 = None,save_option=0, title='UAV Maneuvering'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_zlabel('Z (m)', fontsize=10, labelpad=5)
        ax.set_zlim(-3, 3)
        ax.set_ylim(-9, 9)
        ax.set_xlim(-6, 6)
        ax.set_title(title, pad=20, fontsize=15)

        # target landing point
        ax.scatter3D([self.goal_r_I[0]], [self.goal_r_I[1]], [self.goal_r_I[2]], c="r", marker="x")
        
        #plot the final state
        #final_position = self.get_final_position(wing_len=wing_len)
        #c_x, c_y, c_z = final_position[0:3]
        #r1_x, r1_y, r1_z = final_position[3:6]
        #r2_x, r2_y, r2_z = final_position[6:9]
        #r3_x, r3_y, r3_z = final_position[9:12]
        #r4_x, r4_y, r4_z = final_position[12:15]
        #line_arm1, = ax.plot([c_x, r1_x], [c_y, r1_y], [c_z, r1_z], linewidth=2, color='grey', marker='o', markersize=3)
        #line_arm2, = ax.plot([c_x, r2_x], [c_y, r2_y], [c_z, r2_z], linewidth=2, color='grey', marker='o', markersize=3)
        #line_arm3, = ax.plot([c_x, r3_x], [c_y, r3_y], [c_z, r3_z], linewidth=2, color='grey', marker='o', markersize=3)
        #line_arm4, = ax.plot([c_x, r4_x], [c_y, r4_y], [c_z, r4_z], linewidth=2, color='grey', marker='o', markersize=3)
        
        # plot gate
        if point1 is not None:
            ax.plot([point1[0],point2[0]],[point1[1],point2[1]],[point1[2],point2[2]],linewidth=1,color='red',linestyle='--')
            ax.plot([point2[0],point3[0]],[point2[1],point3[1]],[point2[2],point3[2]],linewidth=1,color='red',linestyle='--')
            ax.plot([point3[0],point4[0]],[point3[1],point4[1]],[point3[2],point4[2]],linewidth=1,color='red',linestyle='--')
            ax.plot([point4[0],point1[0]],[point4[1],point1[1]],[point4[2],point1[2]],linewidth=1,color='red',linestyle='--')
        # data
        position = self.get_quadrotor_position(wing_len, state_traj)
        sim_horizon = np.size(position, 0)

        if state_traj_ref is None:
            position_ref = self.get_quadrotor_position(0, numpy.zeros_like(position))
        else:
            position_ref = self.get_quadrotor_position(wing_len, state_traj_ref)

        ## plot the process of moving window and quadrotor
        #for i in range(10):
        #    a = i*6
        #    b = 0.9-0.1*i
        #    c = (b,b,b)
        #    c_x, c_y, c_z = position[a,0:3]
        #    r1_x, r1_y, r1_z = position[a,3:6]
        #    r2_x, r2_y, r2_z = position[a,6:9]
        #    r3_x, r3_y, r3_z = position[a,9:12]
        #    r4_x, r4_y, r4_z = position[a,12:15]
        #    line_arm1, = ax.plot([c_x, r1_x], [c_y, r1_y], [c_z, r1_z], linewidth=2, color=c, marker='o', markersize=3)
        #    line_arm2, = ax.plot([c_x, r2_x], [c_y, r2_y], [c_z, r2_z], linewidth=2, color=c, marker='o', markersize=3)
        #    line_arm3, = ax.plot([c_x, r3_x], [c_y, r3_y], [c_z, r3_z], linewidth=2, color=c, marker='o', markersize=3)
        #    line_arm4, = ax.plot([c_x, r4_x], [c_y, r4_y], [c_z, r4_z], linewidth=2, color=c, marker='o', markersize=3)

        #    p1_x, p1_y, p1_z = gate_traj1[a, 0,:]
        #    p2_x, p2_y, p2_z = gate_traj1[a, 1,:]
        #    p3_x, p3_y, p3_z = gate_traj1[a, 2,:]
        #    p4_x, p4_y, p4_z = gate_traj1[a, 3,:]
        #    gate_l1, = ax.plot([p1_x,p2_x],[p1_y,p2_y],[p1_z,p2_z],linewidth=1,color=c,linestyle='--')
        #    gate_l2, = ax.plot([p2_x,p3_x],[p2_y,p3_y],[p2_z,p3_z],linewidth=1,color=c,linestyle='--')
        #    gate_l3, = ax.plot([p3_x,p4_x],[p3_y,p4_y],[p3_z,p4_z],linewidth=1,color=c,linestyle='--')
        #    gate_l4, = ax.plot([p4_x,p1_x],[p4_y,p1_y],[p4_z,p1_z],linewidth=1,color=c,linestyle='--')
        

        ## animation
        # gate
        if gate_traj1 is not None:
            p1_x, p1_y, p1_z = gate_traj1[0, 0,:]
            p2_x, p2_y, p2_z = gate_traj1[0, 1,:]
            p3_x, p3_y, p3_z = gate_traj1[0, 2,:]
            p4_x, p4_y, p4_z = gate_traj1[0, 3,:]
            gate_l1, = ax.plot([p1_x,p2_x],[p1_y,p2_y],[p1_z,p2_z],linewidth=1,color='red',linestyle='--')
            gate_l2, = ax.plot([p2_x,p3_x],[p2_y,p3_y],[p2_z,p3_z],linewidth=1,color='red',linestyle='--')
            gate_l3, = ax.plot([p3_x,p4_x],[p3_y,p4_y],[p3_z,p4_z],linewidth=1,color='red',linestyle='--')
            gate_l4, = ax.plot([p4_x,p1_x],[p4_y,p1_y],[p4_z,p1_z],linewidth=1,color='red',linestyle='--')

            #p1_xa, p1_ya, p1_za = gate_traj2[0, 0,:]
            #p2_xa, p2_ya, p2_za = gate_traj2[0, 1,:]
            #p3_xa, p3_ya, p3_za = gate_traj2[0, 2,:]
            #p4_xa, p4_ya, p4_za = gate_traj2[0, 3,:]
            #gate_l1a, = ax.plot([p1_xa,p2_xa],[p1_ya,p2_ya],[p1_za,p2_za],linewidth=1,color='red',linestyle='--')
            #gate_l2a, = ax.plot([p2_xa,p3_xa],[p2_ya,p3_ya],[p2_za,p3_za],linewidth=1,color='red',linestyle='--')
            #gate_l3a, = ax.plot([p3_xa,p4_xa],[p3_ya,p4_ya],[p3_za,p4_za],linewidth=1,color='red',linestyle='--')
            #gate_l4a, = ax.plot([p4_xa,p1_xa],[p4_ya,p1_ya],[p4_za,p1_za],linewidth=1,color='red',linestyle='--')    

        # quadrotor
        line_traj, = ax.plot(position[:1, 0], position[:1, 1], position[:1, 2])
        c_x, c_y, c_z = position[0, 0:3]
        r1_x, r1_y, r1_z = position[0, 3:6]
        r2_x, r2_y, r2_z = position[0, 6:9]
        r3_x, r3_y, r3_z = position[0, 9:12]
        r4_x, r4_y, r4_z = position[0, 12:15]
        line_arm1, = ax.plot([c_x, r1_x], [c_y, r1_y], [c_z, r1_z], linewidth=2, color='red', marker='o', markersize=3)
        line_arm2, = ax.plot([c_x, r2_x], [c_y, r2_y], [c_z, r2_z], linewidth=2, color='blue', marker='o', markersize=3)
        line_arm3, = ax.plot([c_x, r3_x], [c_y, r3_y], [c_z, r3_z], linewidth=2, color='orange', marker='o', markersize=3)
        line_arm4, = ax.plot([c_x, r4_x], [c_y, r4_y], [c_z, r4_z], linewidth=2, color='green', marker='o', markersize=3)

        line_traj_ref, = ax.plot(position_ref[:1, 0], position_ref[:1, 1], position_ref[:1, 2], color='gray', alpha=0.5)
        c_x_ref, c_y_ref, c_z_ref = position_ref[0, 0:3]
        r1_x_ref, r1_y_ref, r1_z_ref = position_ref[0, 3:6]
        r2_x_ref, r2_y_ref, r2_z_ref = position_ref[0, 6:9]
        r3_x_ref, r3_y_ref, r3_z_ref = position_ref[0, 9:12]
        r4_x_ref, r4_y_ref, r4_z_ref = position_ref[0, 12:15]
        line_arm1_ref, = ax.plot([c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref], [c_z_ref, r1_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)
        line_arm2_ref, = ax.plot([c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref], [c_z_ref, r2_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)
        line_arm3_ref, = ax.plot([c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref], [c_z_ref, r3_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)
        line_arm4_ref, = ax.plot([c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref], [c_z_ref, r4_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)

        # time label
        time_template = 'time = %.2fs'
        time_text = ax.text2D(0.66, 0.55, "time", transform=ax.transAxes)

        # customize
        if state_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['learned', 'OC solver'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))

        def update_traj(num):
            # customize
            time_text.set_text(time_template % (num * dt))

            # trajectory
            line_traj.set_data(position[:num, 0], position[:num, 1])
            line_traj.set_3d_properties(position[:num, 2])


            # uav
            c_x, c_y, c_z = position[num, 0:3]
            r1_x, r1_y, r1_z = position[num, 3:6]
            r2_x, r2_y, r2_z = position[num, 6:9]
            r3_x, r3_y, r3_z = position[num, 9:12]
            r4_x, r4_y, r4_z = position[num, 12:15]

            line_arm1.set_data_3d([c_x, r1_x], [c_y, r1_y],[c_z, r1_z])
            #line_arm1.set_3d_properties()

            line_arm2.set_data_3d([c_x, r2_x], [c_y, r2_y],[c_z, r2_z])
            #line_arm2.set_3d_properties()

            line_arm3.set_data_3d([c_x, r3_x], [c_y, r3_y],[c_z, r3_z])
            #line_arm3.set_3d_properties()

            line_arm4.set_data_3d([c_x, r4_x], [c_y, r4_y],[c_z, r4_z])
            #line_arm4.set_3d_properties()

            # trajectory ref
            nu=sim_horizon-1
            line_traj_ref.set_data_3d(position_ref[:nu, 0], position_ref[:nu, 1],position_ref[:nu, 2])
            #line_traj_ref.set_3d_properties()

            # uav ref
            c_x_ref, c_y_ref, c_z_ref = position_ref[nu, 0:3]
            r1_x_ref, r1_y_ref, r1_z_ref = position_ref[nu, 3:6]
            r2_x_ref, r2_y_ref, r2_z_ref = position_ref[nu, 6:9]
            r3_x_ref, r3_y_ref, r3_z_ref = position_ref[nu, 9:12]
            r4_x_ref, r4_y_ref, r4_z_ref = position_ref[nu, 12:15]

            line_arm1_ref.set_data_3d([c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref],[c_z_ref, r1_z_ref])
            #line_arm1_ref.set_3d_properties()

            line_arm2_ref.set_data_3d([c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref],[c_z_ref, r2_z_ref])
            #line_arm2_ref.set_3d_properties()

            line_arm3_ref.set_data_3d([c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref],[c_z_ref, r3_z_ref])
            #line_arm3_ref.set_3d_properties()

            line_arm4_ref.set_data_3d([c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref],[c_z_ref, r4_z_ref])
            #line_arm4_ref.set_3d_properties()

            ## plot moving gate
            if gate_traj1 is not None:
                p1_x, p1_y, p1_z = gate_traj1[num, 0,:]
                p2_x, p2_y, p2_z = gate_traj1[num, 1,:]
                p3_x, p3_y, p3_z = gate_traj1[num, 2,:]
                p4_x, p4_y, p4_z = gate_traj1[num, 3,:]       

                gate_l1.set_data_3d([p1_x,p2_x],[p1_y,p2_y],[p1_z,p2_z])
                gate_l2.set_data_3d([p2_x,p3_x],[p2_y,p3_y],[p2_z,p3_z]) 
                gate_l3.set_data_3d([p3_x,p4_x],[p3_y,p4_y],[p3_z,p4_z]) 
                gate_l4.set_data_3d([p4_x,p1_x],[p4_y,p1_y],[p4_z,p1_z])


                #p1_xa, p1_ya, p1_za = gate_traj2[num, 0,:]
                #p2_xa, p2_ya, p2_za = gate_traj2[num, 1,:]
                #p3_xa, p3_ya, p3_za = gate_traj2[num, 2,:]
                #p4_xa, p4_ya, p4_za = gate_traj2[num, 3,:]       

                #gate_l1a.set_data_3d([p1_xa,p2_xa],[p1_ya,p2_ya],[p1_za,p2_za])
                #gate_l2a.set_data_3d([p2_xa,p3_xa],[p2_ya,p3_ya],[p2_za,p3_za]) 
                #gate_l3a.set_data_3d([p3_xa,p4_xa],[p3_ya,p4_ya],[p3_za,p4_za]) 
                #gate_l4a.set_data_3d([p4_xa,p1_xa],[p4_ya,p1_ya],[p4_za,p1_za])




                return line_traj,gate_l1,gate_l2,gate_l3,gate_l4,line_arm1, line_arm2, line_arm3, line_arm4, \
                    line_traj_ref, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref, time_text

            return line_traj, line_arm1, line_arm2, line_arm3, line_arm4, \
                line_traj_ref, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=dt*1000, blit=True)

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('case2'+title + '.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()

    def plot_position(self,state_traj,dt = 0.1):
        fig, axs = plt.subplots(3)
        fig.suptitle('position vs t')
        N = len(state_traj[:,0])
        x = np.arange(0,N*dt,dt)
        axs[0].plot(x,state_traj[:,0])
        axs[1].plot(x,state_traj[:,1])
        axs[2].plot(x,state_traj[:,2])
        plt.show()
        
    def plot_velocity(self,state_traj,dt = 0.1):
        fig, axs = plt.subplots(3)
        fig.suptitle('velocity vs t')
        N = len(state_traj[:,0])
        x = np.arange(0,N*dt,dt)
        axs[0].plot(x,state_traj[:,3])
        axs[1].plot(x,state_traj[:,4])
        axs[2].plot(x,state_traj[:,5])
        plt.show()

    def plot_quaternions(self,state_traj,dt = 0.1):
        fig, axs = plt.subplots(4)
        fig.suptitle('quaternions vs t')
        N = len(state_traj[:,0])
        x = np.arange(0,N*dt,dt)
        axs[0].plot(x,state_traj[:,6])
        axs[1].plot(x,state_traj[:,7])
        axs[2].plot(x,state_traj[:,8])
        axs[3].plot(x,state_traj[:,9])
        plt.show()
    
    def plot_angularrate(self,state_traj,dt = 0.1):
        plt.title('angularrate vs time')
        N = len(state_traj[:,0])
        x = np.arange(0,N*dt,dt)
        plt.plot(x,state_traj[:,10],color = 'b', label = 'w1')
        plt.plot(x,state_traj[:,11],color = 'r', label = 'w2')
        plt.plot(x,state_traj[:,12],color = 'y', label = 'w3')
        plt.xlabel('t')
        plt.ylabel('w')
        plt.grid(True,color='0.6',dashes=(2,2,1,1))
        plt.legend()
        plt.show()
        plt.savefig('./angularrate.png')

    def plot_input(self,control_traj,dt = 0.1):
        N = int(len(control_traj[:,0]))
        x = np.arange(0,round(N*dt,1),dt)
        plt.plot(x,control_traj[:,0],color = 'b', label = 'u1')
        plt.plot(x,control_traj[:,1],color = 'r', label = 'u2')
        plt.plot(x,control_traj[:,2],color = 'y', label = 'u3')
        plt.plot(x,control_traj[:,3],color = 'g', label = 'u4')
        plt.title('input vs time')
        plt.ylim([0,3])
        plt.xlabel('t')
        plt.ylabel('u')
        plt.grid(True,color='0.6',dashes=(2,2,1,1))
        plt.legend()
        plt.show()



    def dir_cosine(self, q):
        C_B_I = vertcat(
            horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
        return C_B_I

    def skew(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2], v[1]),
            horzcat(v[2], 0, -v[0]),
            horzcat(-v[1], v[0], 0)
        )
        return v_cross

    def omega(self, w):
        omeg = vertcat(
            horzcat(0, -w[0], -w[1], -w[2]),
            horzcat(w[0], 0, w[2], -w[1]),
            horzcat(w[1], -w[2], 0, w[0]),
            horzcat(w[2], w[1], -w[0], 0)
        )
        return omeg

    def quaternion_mul(self, p, q):
        return vertcat(p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                       p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                       p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                       p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
                       )

## define the class of the gate (kinematics)
class gate:
    ## using 12 coordinates to define a gate
    def __init__(self, gate_point = None):
        self.gate_point = gate_point

        ##obtain the position (centroid)
        self.centroid = np.array([np.mean(self.gate_point[:,0]),np.mean(self.gate_point[:,1]),np.mean(self.gate_point[:,2])])

        ## obtain the orientation
        az = norm(np.array([0,0,1]))
        ay = norm(np.cross(self.gate_point[1]-self.gate_point[0],self.gate_point[2]-self.gate_point[1]))
        ax = np.cross(ay,az)
        self.ay = ay
        self.I_G = np.array([ax,ay,az])

    ## rotate an angle around y axis of thw window
    def rotate_y(self,angle):
        ## define the rotation matrix to rotate
        rotation = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        gate_point = self.gate_point - np.array([self.centroid,self.centroid,self.centroid,self.centroid])
        for i in range(4):
            [gate_point[i,0],gate_point[i,2]] = np.matmul(rotation,np.array([gate_point[i,0],gate_point[i,2]]))
        self.gate_point = gate_point + np.array([self.centroid,self.centroid,self.centroid,self.centroid])

        ## update the orientation and the position
        self.centroid = np.array([np.mean(self.gate_point[:,0]),np.mean(self.gate_point[:,1]),np.mean(self.gate_point[:,2])])
        az = norm(np.array([0,0,1]))
        ay = norm(np.cross(self.gate_point[1]-self.gate_point[0],self.gate_point[2]-self.gate_point[1]))
        ax = np.cross(ay,az)
        self.ay = ay
        self.I_G = np.array([ax,ay,az])

    ## rotate an angle around z axis of thw window
    def rotate(self,angle):
        ## define the rotation matrix to rotate
        rotation = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        gate_point = self.gate_point - np.array([self.centroid,self.centroid,self.centroid,self.centroid])
        for i in range(4):
            gate_point[i,0:2] = np.matmul(rotation,gate_point[i,0:2])
        self.gate_point = gate_point + np.array([self.centroid,self.centroid,self.centroid,self.centroid])

        ## update the orientation and the position
        self.centroid = np.array([np.mean(self.gate_point[:,0]),np.mean(self.gate_point[:,1]),np.mean(self.gate_point[:,2])])
        az = norm(np.array([0,0,1]))
        ay = norm(np.cross(self.gate_point[1]-self.gate_point[0],self.gate_point[2]-self.gate_point[1]))
        ax = np.cross(ay,az)
        self.ay = ay
        self.I_G = np.array([ax,ay,az])

    ## translate the gate in world frame
    def translate(self,displace):
        self.gate_point = self.gate_point + np.array([displace,displace,displace,displace])
        self.centroid = np.array([np.mean(self.gate_point[:,0]),np.mean(self.gate_point[:,1]),np.mean(self.gate_point[:,2])])

        ## update the orientation and the positio
        az = norm(np.array([0,0,1]))
        ay = norm(np.cross(self.gate_point[1]-self.gate_point[0],self.gate_point[2]-self.gate_point[1]))
        ax = np.cross(ay,az)
        self.ay = ay
        self.I_G = np.array([ax,ay,az])

    ## 'out' means return the 12 coordinates of the gate
    def translate_out(self,displace):
        return self.gate_point + np.array([displace,displace,displace,displace])

    def rotate_y_out(self,angle):
        rotation = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        gate_point = self.gate_point - np.array([self.centroid,self.centroid,self.centroid,self.centroid])
        for i in range(4):
            [gate_point[i,0],gate_point[i,2]] = np.matmul(rotation,np.array([gate_point[i,0],gate_point[i,2]]))
        gate_point = gate_point + np.array([self.centroid,self.centroid,self.centroid,self.centroid])
        return gate_point

    def rotate_out(self,angle):
        rotation = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        gate_point = self.gate_point - np.array([self.centroid,self.centroid,self.centroid,self.centroid])
        for i in range(4):
            gate_point[i,0:2] = np.matmul(rotation,gate_point[i,0:2])
        gate_point = gate_point + np.array([self.centroid,self.centroid,self.centroid,self.centroid])
        return gate_point

    ## given time horizon T and time interval dt, return a sequence of position representing the random move of the gate
    def random_move(self, T = 5, dt = 0.01):
        gate_point = self.gate_point
        move = [gate_point]
        ## initial random velocity
        velo = np.random.normal(0,0.2,size=2)
        for i in range(int(T/dt)):
            ## random acceleration
            accel = np.random.normal(0,2,size=2)
            ## integration
            velo += dt*accel
            velocity = np.clip(np.array([velo[0],0,velo[1]]),-0.4,0.4)
            for j in range(4):
                gate_point[j] += dt * velocity
            move = np.concatenate((move,[gate_point]),axis=0)
        return move
    
    ## given constant velocity and angular velocity around y axis, return a sequence of position representing the random move of the gate 
    def move(self, T = 5, dt = 0.01, v = [0,0,0], w = 0):
        gate_point = self.gate_point
        move = [gate_point]
        velo = np.array(v)
        # define the rotation matrix
        rotation = np.array([[math.cos(dt*w),-math.sin(dt*w)],[math.sin(dt*w),math.cos(dt*w)]])
        for i in range(int(T/dt)):
            v_noise = np.clip(np.random.normal(0,0.1,3),-0.2,0.2)
            centroid = np.array([np.mean(gate_point[:,0]),np.mean(gate_point[:,1]),np.mean(gate_point[:,2])])
            gate_pointx = gate_point - np.array([centroid,centroid,centroid,centroid])
            # rotation
            for i in range(4):
                [gate_pointx[i,0],gate_pointx[i,2]] = np.matmul(rotation,np.array([gate_pointx[i,0],gate_pointx[i,2]]))
            gate_point = gate_pointx + np.array([centroid,centroid,centroid,centroid])
            # translation
            for j in range(4):
                gate_point[j] += dt * (velo+v_noise)
            move = np.concatenate((move,[gate_point]),axis=0)
        return move
    
    ## transform the state in world frame to the state in window frame
    def transform(self, inertial_state):
        outputs = np.zeros(13)
        ## position
        outputs[0:3] = np.matmul(self.I_G, inertial_state[0:3] - self.centroid)
        ## velocity
        outputs[3:6] = np.matmul(self.I_G, inertial_state[3:6])
        ## angular velocity
        outputs[10:13] = inertial_state[10:13]
        ## attitude
        quat = np.zeros(4)
        quat[0:3] = inertial_state[7:10]
        quat[3] = inertial_state[6]
        r1 = R.from_quat(quat)
        # attitude transformation
        r2 = R.from_matrix(np.matmul(self.I_G,r1.as_matrix()))
        quat_out = np.array(r2.as_quat())
        outputs[6] = quat_out[3]
        outputs[7:10] = quat_out[0:3]
        return outputs

    ## transform the final point in world frame to the point in window frame
    def t_final(self, final_point):
        return np.matmul(self.I_G, final_point - self.centroid)
        

def toQuaternion(angle, dir):
    if type(dir) == list:
        dir = numpy.array(dir)
    dir = dir / numpy.linalg.norm(dir)
    quat = numpy.zeros(4)
    quat[0] = math.cos(angle / 2)
    quat[1:] = math.sin(angle / 2) * dir
    return quat.tolist()


# normalized verctor
def normalizeVec(vec):
    if type(vec) == list:
        vec = np.array(vec)
    vec = vec / np.linalg.norm(vec)
    return vec


def quaternion_conj(q):
    conj_q = q
    conj_q[1] = -q[1]
    conj_q[2] = -q[2]
    conj_q[3] = -q[3]
    return conj_q