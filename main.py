## this file is for traversing moving narrow window


from quad_model import *
from quad_policy import *
from quad_nn import *
from quad_moving import *
# initialization

#gate_point = np.array([[-0.55,0,1],[0.55,0,1],[0.55,0,-1],[-0.55,0,-1]])
#gate1 = gate(gate_point)

# sample the input
# Init_pos   = np.load('start_point.npy')
# Final_pos  = np.load('Final_point.npy')
# Init_angle = np.load('initial_angle.npy')
# inputs = nn_sample(init_pos=Init_pos,final_pos=Final_pos,init_angle=Init_angle)
inputs = nn_sample()
start_point = inputs[0:3]
final_point = inputs[3:6]
# np.save('start_point',start_point)
# np.save('Final_point',final_point)
# np.save('initial_angle',inputs[8])
# initial obstacle
gate_point0 = np.array([[-inputs[7]/2,0,1],[inputs[7]/2,0,1],[inputs[7]/2,0,-1],[-inputs[7]/2,0,-1]])
gate1 = gate(gate_point0)
g_init_p = inputs[8]
gate1.rotate_y(inputs[8])
gate_point = gate1.gate_point
gate1 = gate(gate_point)

# initial traversal problem
quad1 = run_quad(goal_pos=inputs[3:6],ini_r=inputs[0:3].tolist(),ini_q=toQuaternion(inputs[6],[0,0,1]))
quad1.init_obstacle(gate_point.reshape(12))
quad1.uav1.setDyn(0.01)

ini_state = np.array(quad1.ini_state)

horizon = 50

FILE = "nn3_1.pth"
model = torch.load(FILE)

## define the kinematics of the narrow window
v = np.array([1,0.3,0.4])
w = pi/2
gate_move, V = gate1.move(v = v ,w = w)

# initialization
state = quad1.ini_state # state= feedback from pybullet, 13-by-1, 3 position, 3 velocity (world frame), 4 quaternion, 3 angular rate
u = [0,0,0,0]
tm = [0,0,0,0]
state_n = [state]
control_n = [u]
control_tm = [tm]
hl_para = [0,0,0,0,0,0,0]
hl_variable = [hl_para]
gate_n = gate(gate_move[0])
t_guess = magni(gate_n.centroid-state[0:3])/3
Ttra    = []
T       = []
Time    = []
Pitch   = []
j = 0
for i in range(500):
    gate_n = gate(gate_move[i])
    t = solver(model,state,final_point,gate_n,V[i],w)
    t_tra = t+i*0.01
    gap_pitch = g_init_p + w*i*0.01
    print('step',i,'tranversal time=',t,'gap_pitch=',gap_pitch*180/pi)
    # print('step',i,'abs_tranversal time=',t_tra)
    Ttra = np.concatenate((Ttra,[t_tra]),axis = 0)
    T = np.concatenate((T,[t]),axis = 0)
    Time = np.concatenate((Time,[i*0.01]),axis = 0)
    Pitch = np.concatenate((Pitch,[gap_pitch]),axis = 0)
    if (i%10)==0: # control frequency = 10 hz
    ## obtain the current gate state
    ## solve for the traversal time
        # t = solver(model,state,final_point,gate_n,V[i],w)
        # t_tra = t+i*0.01
        # print('step',i,'tranversal time=',t)
        # print('step',i,'abs_tranversal time=',t_tra)
        # Ttra = np.concatenate((Ttra,[t_tra]),axis = 0)
    #print(t,' ',i)
    ## obtain the future traversal window state
        gate_n.translate(t*V[i])
        gate_n.rotate_y(t*w)
        # print('rotation matrix I_G=',gate_n.I_G)
    ## obtain the state in window frame 
        inputs = np.zeros(18)
        inputs[16] = magni(gate_n.gate_point[0,:]-gate_n.gate_point[1,:])
        inputs[17] = atan((gate_n.gate_point[0,2]-gate_n.gate_point[1,2])/(gate_n.gate_point[0,0]-gate_n.gate_point[1,0])) # compute the actual gate pitch ange in real-time
        inputs[0:13] = gate_n.transform(state)
        inputs[13:16] = gate_n.t_final(final_point)
    
        out = model(inputs).data.numpy()
        print('tra_position=',out[0:3],'tra_time_dnn2=',out[6])
    #print(out)
        # if (horizon-1*i/10) <= 30:
        #     Horizon =30
        # else:
        #     Horizon = int(horizon-1*i/10)

    ## solve the mpc problem and get the control command
        quad2 = run_quad(goal_pos=inputs[13:16],horizon =50)
        u = quad2.get_input(inputs[0:13],u,out[0:3],out[3:6],out[6]) # control input 4-by-1 thrusts to pybullet
        j += 1
    state = np.array(quad1.uav1.dyn_fn(state, u)).reshape(13) # Yixiao's simulation environment ('uav1.dyn_fn'), replaced by pybullet
    state_n = np.concatenate((state_n,[state]),axis = 0)
    control_n = np.concatenate((control_n,[u]),axis = 0)
    u_m = quad1.uav1.u_m
    u1 = np.reshape(u,(4,1))
    tm = np.matmul(u_m,u1)
    tm = np.reshape(tm,4)
    control_tm = np.concatenate((control_tm,[tm]),axis = 0)
    hl_variable = np.concatenate((hl_variable,[out]),axis=0)
np.save('gate_move_traj',gate_move)
np.save('uav_traj',state_n)
np.save('uav_ctrl',control_n)
np.save('abs_tra_time',Ttra)
np.save('tra_time',T)
np.save('Time',Time)
np.save('Pitch',Pitch)
np.save('HL_Variable',hl_variable)
quad1.uav1.play_animation(wing_len=1.5,gate_traj1=gate_move ,state_traj=state_n)
# quad1.uav1.plot_input(control_n)
# quad1.uav1.plot_angularrate(state_n)
# quad1.uav1.plot_T(control_tm)
# quad1.uav1.plot_M(control_tm)
