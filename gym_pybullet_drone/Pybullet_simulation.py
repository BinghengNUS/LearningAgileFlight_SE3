"""
Pybullet simulation code of YiXiao's research based on fly.py.
Written by: ZSN

Version: 2.0
"""

from re import T
import time
import numpy as np
import pybullet as p

import sys
sys.path.append('../')

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

from GateAviary import GateAviary, DrawingCommander
from Yixiao_ctrl_wrapper import YXCtrlWrapper


# parameters
DEFAULT_DRONES = DroneModel("hb")  # our model
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = True  # to add RPM sliders and UAV frame
DEFAULT_AGGREGATE = False
DEFAULT_OBSTACLES = False
DEFAULT_OUTPUT_FOLDER = 'Pybullet records'
# trained model file
MODEL_FILE = "nn3_1.pth"
# debug switches
DEFAULT_SHOW_GATE = True
DEFAULT_SHOW_TRJ = True
DEFAULT_SHOW_SPEED = True
# simulation settings
DEFAULT_SIMULATION_FREQ_HZ = 100
DEFAULT_CONTROL_FREQ_HZ = 10
DEFAULT_DURATION_SEC = 5
# gate paras
DEFAULT_GATE_S_ORI = np.array([0, 0, 3])  # starting gate origin
DEFAULT_GATE_PARAS = dict(start_p=-3, st_p_range=2, end_p=4, end_p_range=1,
                          gate_wid_rand=[0.35, 0.1], gate_wid_lim=[0.3, 0.4])
DEFAULT_HALF_GATE_HEI = 0.5 # 0.4 for the dynamic gap
DEFAULT_GATE_V = np.array([1.0, 0.3, 0.4])
DEFAULT_GATE_W = 1/2*np.pi
# video capture options (for recording)
USE_LAST_SIM_SETTING = True  # reuse last random settings when finding a good simulation
DRONE_FRONT_VIEW = False  # whether to use front view of drone, or the full view
STARTING_PAUSE_SEC = 1  # shortly pause for preparation
MANUAL_SYNC_STEP = 0.2  # set to 0.8 or more when recording; avoid timeline ununiformity
SHOW_TRAVERSAL_GATE = True  # switches
SHOW_UAV_IN_PROCESS = True

def run_simulation(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        aggregate=DEFAULT_AGGREGATE,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        show_gate=DEFAULT_SHOW_GATE,
        show_trajectory=DEFAULT_SHOW_TRJ,
        show_speed=DEFAULT_SHOW_SPEED,
        gate_s_ori=DEFAULT_GATE_S_ORI,
        gate_paras=DEFAULT_GATE_PARAS,
        half_gt_hei=DEFAULT_HALF_GATE_HEI,
        gate_v=DEFAULT_GATE_V,
        gate_w=DEFAULT_GATE_W,
        model_file=MODEL_FILE,
        use_last_sim_setting=USE_LAST_SIM_SETTING,
        drone_front_view=DRONE_FRONT_VIEW,
        starting_pause=STARTING_PAUSE_SEC,
        manual_sync_step=MANUAL_SYNC_STEP,
        show_traversal_gate=SHOW_TRAVERSAL_GATE,
        show_uav_in_process=SHOW_UAV_IN_PROCESS,
):
    CTRL_EVERY_N_STEPS = int(np.floor(simulation_freq_hz / control_freq_hz))

    #### Initialize the controllers ############################
    ctrl = [YXCtrlWrapper(drone_model=drone,
                          gate_paras=gate_paras,
                          half_gt_hei=half_gt_hei,
                          gate_v=gate_v,
                          gate_w=gate_w,
                          relative_ori=gate_s_ori,
                          model_file=model_file,
                          ctrl_every_n_steps=CTRL_EVERY_N_STEPS,
                          replicate_sim=use_last_sim_setting,
                          ) for i in range(num_drones)]
    gate_move = ctrl[0].gate_move + gate_s_ori

    #### Initialize the simulation #############################
    INIT_XYZS = np.array(
        [ctrl[0].start_point + gate_s_ori for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0, ctrl[0].UAV_ini_yaw] for i in range(num_drones)])
    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1

    #### Create the environment ##
    env = GateAviary(drone_model=drone,
                     num_drones=num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=physics,
                     neighbourhood_radius=10,
                     freq=simulation_freq_hz,
                     aggregate_phy_steps=AGGR_PHY_STEPS,
                     gui=gui,
                     record=record_video,
                     obstacles=obstacles,
                     user_debug_gui=user_debug_gui,
                     # new paras below
                     show_gate=show_gate,
                     show_trajectory=show_trajectory,
                     show_speed=show_speed,
                     uav_final_point=ctrl[0].final_point + gate_s_ori,
                     relative_ori=gate_s_ori,
                     drone_front_view=drone_front_view,
                     gate_width=ctrl[0].gate_width,  # new below in version 2.0
                     gate_height=half_gt_hei * 2,
                     gate_ori_pitch=ctrl[0].gate_pitch,
                     gate_v=gate_v,
                     gate_w=gate_w,
                     manual_sync_step=manual_sync_step,
                     )

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(simulation_freq_hz / AGGR_PHY_STEPS),
                    num_drones=num_drones,
                    output_folder=output_folder,
                    )

    #### Run the simulation ####################################
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(num_drones)}
    START = time.time()
    gate_traversal = False  # control drawing of traversal gate using 2 states
    already_drawn = False
    drawing_commander = DrawingCommander()  # control drawing of UAV in-process

    time.sleep(starting_pause)
    for i in range(0, int(duration_sec * env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Step the simulation ###################################
        gate_points = gate_move[i, :, :]
        obs, reward, done, info = env.step(action=action,
                                           gate_points=gate_points,
                                           # to control whether show traversal gate
                                           gate_traversal=show_traversal_gate and gate_traversal and not already_drawn,
                                           draw_drone=show_uav_in_process and drawing_commander.command,
                                           )
        if gate_traversal:  # only draw once
            gate_traversal = False
            already_drawn = True

        state = obs[str(0)]["state"]  # V2: state every step

        #### Compute control at the desired frequency ##############
        if i % CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            for j in range(num_drones):
                action[str(j)], t = ctrl[j].computeControl(
                    control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                    cur_pos=state[0:3],
                    cur_quat=state[3:7],
                    cur_vel=state[10:13],
                    cur_ang_vel=state[13:16],
                    target_pos=None,
                    )

        # V2 traversal gate: use position condition
        gate_pos, _ = p.getBasePositionAndOrientation(env.GATE_ID, physicsClientId=env.CLIENT)
        if state[1] > gate_pos[1]-0.3:  # -0.3 exceeds gate y position
            gate_traversal = True

        drawing_commander.update_command(t)  # update every single step

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i / env.SIM_FREQ,
                       state=obs[str(j)]["state"],
                       )

        #### Printout ##############################################
        if i % env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("YiXiao")  # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()


if __name__ == '__main__':
    run_simulation()
