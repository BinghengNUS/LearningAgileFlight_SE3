"""
2.0 version of GateAviary using new gate visualization method.
Written by: ZSN
"""

import os
import numpy as np
import pybullet as p
import time

import sys
sys.path.append('../')

from gym_pybullet_drones.envs.DynAviary import DynAviary  # to give input as thrust and torques
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class GateAviary(DynAviary):
    """Sub-class of control aviary visualising a rotating narrow gate."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 1,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results',
                 show_gate=True,  # new - show the narrow gate
                 show_trajectory=True,
                 show_speed=True,
                 uav_final_point=None,  # new - for drawing trajectory
                 relative_ori=None,
                 drone_front_view=False,
                 gate_width=None,  # new - version 2.0
                 gate_height=None,
                 gate_ori_pitch=None,
                 gate_v=None,
                 gate_w=None,
                 manual_sync_step=0,
                 ):

        self.SHOW_GATE = show_gate
        self.SHOW_TRAJECTORY = show_trajectory
        self.SHOW_SPEED = show_speed

        self.UAV_FINAL_POINT = uav_final_point
        self.RELATIVE_ORI = relative_ori
        self.DRONE_FRONT_VIEW = drone_front_view

        # new for version 2.0
        MODEL = 'model/window'  # general name for gate models
        self.gate_ori_pitch = gate_ori_pitch
        self.gate_v = gate_v
        self.gate_w = gate_w
        self.gate_height = gate_height  # for display on frozen gate
        self.DRONE_FROZEN = 'model/hb.urdf'
        self.manual_sync_tim = manual_sync_step

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder,
                         )

        # load gate model
        if show_gate:
            model, self.MODEL_FROZEN = scaled_model(MODEL, gate_width, gate_height)
            self.GATE_ID = p.loadURDF(model,
                                      relative_ori,
                                      p.getQuaternionFromEuler([gate_ori_pitch, 0, -np.pi/2]),
                                      physicsClientId=self.CLIENT,
                                      )

        if gui and not drone_front_view:
            p.resetDebugVisualizerCamera(  # full view (in max window)
                                         cameraDistance=np.linalg.norm((uav_final_point - initial_xyzs) / 1.5),
                                         cameraYaw=-35,
                                         cameraPitch=-15,
                                         cameraTargetPosition=np.mean(
                                             np.vstack((initial_xyzs,
                                                        uav_final_point)), axis=0),  # average of points
                                         physicsClientId=self.CLIENT
                                         )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print(">> Reset: viewMatrix", ret[2])
            print(">> Reset: projectionMatrix", ret[3])

    ################################################################################

    def _housekeeping(self):
        super(GateAviary, self)._housekeeping()
        self.GATE_CENTROID = -1
        self.SPEED_TEXT = -1

        p.changeVisualShape(self.PLANE_ID,
                    linkIndex=-1,
                    rgbaColor=[0] * 4,
                    )
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                           0,
                           rgbBackground=[1, 1, 1],  # BG color
                           physicsClientId=self.CLIENT)

        if self.SHOW_TRAJECTORY:
            # draw trajectory endpoints & starting gate origin
            p.addUserDebugPoints(pointPositions=np.vstack((self.INIT_XYZS,
                                                           self.UAV_FINAL_POINT,
                                                           self.RELATIVE_ORI)),
                                 pointColorsRGB=[[1, 0, 0], [0, 1, 0], [1, 1, 0]],
                                 pointSize=8,
                                 lifeTime=0,
                                 physicsClientId=self.CLIENT,
                                 )

    ################################################################################

    def step(self,
             action,
             gate_points=None,  # 4x3 gate corner points
             gate_traversal=False,
             draw_drone=False,
             ):
        """
        Adding gate movement, trajectories and other visual effects in display.
        """
        obs, reward, done, info = super(GateAviary, self).step(action=action)
        state = obs[str(0)]["state"]  # extracted for our use

        curr_sim_time = (self.step_counter - 1) * self.TIMESTEP
        curr_gate_pos = self.RELATIVE_ORI + curr_sim_time * self.gate_v
        curr_gate_pitch = self.gate_ori_pitch + curr_sim_time * self.gate_w
        text_size = 2 if self.DRONE_FRONT_VIEW else 1.1

        # show UAV and gate trajectories
        if self.SHOW_TRAJECTORY and self.step_counter % 1 == 0:  # can control display frequency
            p.addUserDebugPoints(pointPositions=[state[0:3],
                                                 np.mean(gate_points, axis=0)],  # gate centroid
                                 pointColorsRGB=[[0, 0, 1], [1, 0.77, 0]],
                                 pointSize=2,
                                 lifeTime=0,
                                 physicsClientId=self.CLIENT,
                                 )
            # show moving gate centroid
            self.GATE_CENTROID = p.addUserDebugPoints([np.mean(gate_points, axis=0)],  # gate centroid
                                                      pointColorsRGB=[[1, 0.55, 0]],
                                                      pointSize=8,
                                                      lifeTime=0,
                                                      replaceItemUniqueId=int(self.GATE_CENTROID),
                                                      physicsClientId=self.CLIENT,
                                                      )
        # show gate
        if self.SHOW_GATE and self.step_counter % 1 == 0:
            # new show gate
            p.resetBasePositionAndOrientation(bodyUniqueId=self.GATE_ID,
                                              posObj=curr_gate_pos,
                                              ornObj=p.getQuaternionFromEuler([curr_gate_pitch, 0, -np.pi/2]),
                                              physicsClientId=self.CLIENT,
                                              )
        # show traversal gate
        if gate_traversal and self.SHOW_GATE:
            # show traversal gate (and drone also)
            p.loadURDF(self.MODEL_FROZEN,
                       curr_gate_pos,
                       p.getQuaternionFromEuler([curr_gate_pitch, 0, -np.pi / 2]),
                       physicsClientId=self.CLIENT,
                       )

            # show pitch angle of gate at traversal
            p.addUserDebugText(f'Gate pitch: {get_gate_pitch(curr_gate_pitch):.2f} deg',
                               # on center and top of frozen gate
                               textPosition=np.hstack((np.mean(gate_points[:, :2], axis=0) + [-0.4, 0.2],
                                                      np.max(gate_points[:, 2]) * 1.04)),
                               textColorRGB=[0, 0.15, 0.45],
                               lifeTime=0,
                               textSize=text_size,
                               physicsClientId=self.CLIENT,
                               )

        # new: show drone in process
        if (draw_drone or gate_traversal) and self.SHOW_TRAJECTORY:
            p.loadURDF(self.DRONE_FROZEN,
                       self.pos[0, :],
                       self.quat[0, :],
                       physicsClientId=self.CLIENT,
                       )

        # show UAV resultant velocity beside the plane
        if self.SHOW_SPEED and self.step_counter % 1 == 0:
            text_position = [-0.1, 0.1, 0.18] if self.DRONE_FRONT_VIEW else [0.25, 0.2, 0.3]
            self.SPEED_TEXT = p.addUserDebugText(f'Speed: {np.linalg.norm(state[10:13]):.2f} m/s',
                                                 textPosition=text_position,
                                                 textColorRGB=[0, 0.1, 0.4],
                                                 lifeTime=0,
                                                 textSize=text_size,
                                                 parentObjectUniqueId=self.DRONE_IDS[0],
                                                 parentLinkIndex=-1,
                                                 replaceItemUniqueId=int(self.SPEED_TEXT),
                                                 physicsClientId=self.CLIENT,
                                                 )
        # change of view when using drone view
        if self.GUI and self.DRONE_FRONT_VIEW:
            # new adjustment method
            p.resetDebugVisualizerCamera(cameraDistance=self.L * 12,
                                         cameraYaw=0,
                                         cameraPitch=-1,
                                         cameraTargetPosition=self.pos[0, :] + np.array([0, self.L * 6, self.L * 1.2]),
                                         physicsClientId=self.CLIENT
                                         )
        # new in V2.0: manual sync
        time.sleep(self.manual_sync_tim)

        return obs, reward, done, info


class DrawingCommander:
    """
    A class for comparing time to control drawing in-process drones.
    Note: must update command in every single simulation step.
    """
    def __init__(self):
        self.pointer = 'undef'
        self.command = False
        self.trigger_time_ser = np.delete(np.arange(3, -21, -1), 3)  # except traversal instant

    def update_command(self, t):
        if self.pointer != 'undef':
            if t <= self.trigger_time_ser[self.pointer]:
                self.pointer += 1
                self.command = True
                return
        else:
            # initialization
            self.pointer = 0
            # to draw a drone when t reaches these time values
            self.trigger_time_ser = self.trigger_time_ser * t * 0.33
        self.command = False


def scaled_model(model_name, width, height):
    """change window size subject to random assignment"""
    model = model_name + '.urdf'
    with open(model, 'r') as f:
        lines = f.readlines()
    for idx in range(len(lines)):
        if lines[idx].strip().startswith('<mesh'):
            break
    line = lines[idx].split('"')

    with open(model, 'w') as f:
        line[-2] = str(np.array([1.1, 0.5, 0.6681318425003423]) * np.array([1, width, height]))[1:-1]  # to real size
        lines[idx] = '"'.join(line)
        f.writelines(lines)

    frozen_model = model_name + '_frozen.urdf'
    with open(frozen_model, 'w') as f:
        line[1] = os.path.split(model_name)[-1] + '_frozen.obj'  # change model
        lines[idx] = '"'.join(line)
        f.writelines(lines)

    return model, frozen_model


def get_gate_pitch(pitch_ini_frame):
    """
    Get gate pitch angle from the visual sense.
    """
    return (pitch_ini_frame / np.pi % 1 - 0.5) * 180
