"""A wrapper for residual learning framework."""
from robopianist.controller.ik_controller import move_finger_to_key, move_fingers_to_keys, move_fingers_to_pos_qp
import collections
from typing import Any, Dict, Optional

import dm_env
import numpy as np
from dm_env import specs
from dm_env_wrappers import EnvironmentWrapper
import math
from dm_control import mjcf
from dm_control.utils.rewards import tolerance
from dm_control.mujoco.wrapper import mjbindings
import random
mjlib = mjbindings.mjlib

_FINGERTIP_CLOSE_ENOUGH = 0.01

class ResidualWrapper(EnvironmentWrapper):
    """Change step function."""
    def __init__(
        self,
        environment: dm_env.Environment,
        demonstrations_lh: np.ndarray, # (600, 8, 6)
        demonstrations_rh: np.ndarray, # (600, 8, 6)
        demo_ctrl_timestep: float = 0.05,
        rsi: bool = False,
        enable_ik: bool = True,
        external_demo: bool = False,
    ) -> None:
        super().__init__(environment)
        self._demonstrations_lh = demonstrations_lh # (600, 8, 6)
        self._demonstrations_rh = demonstrations_rh # (600, 8, 6)
        useful_columns = [0, 1, 2]
        self._demonstrations_lh = self._demonstrations_lh[:, useful_columns, :] # (600, 3, 6)
        self._demonstrations_rh = self._demonstrations_rh[:, useful_columns, :] # (600, 3, 6)
        print(self._demonstrations_lh)
        self._step_scale = self._environment.task.control_timestep / demo_ctrl_timestep
        self._reference_frame_idx = -int(round(self._environment.task._initial_buffer_time/
                                        self._environment.task.control_timestep))
        self._rsi = rsi
        self._enable_ik = enable_ik
        assert self._demonstrations_lh.shape[0] == self._demonstrations_rh.shape[0]
        self._demonstrations_length = self._demonstrations_lh.shape[0] # 600
        # Update the observation spec.
        self._wrapped_observation_spec = self._environment.observation_spec()
        self._observation_spec = collections.OrderedDict()
        self._observation_spec.update(self._wrapped_observation_spec)
        # Add the prior action observation.
        prior_action = np.zeros(self._environment.action_spec().shape[0]-1, dtype=np.float64)
        prior_action_spec = specs.Array(
            shape=prior_action.shape, dtype=prior_action.dtype, name='prior_action'
        )
        self._observation_spec['prior_action'] = prior_action_spec
        # # Add the demo observation.
        # demo_lh= self._demonstrations_lh[0:self.task._n_steps_lookahead+1].flatten()
        # demo_rh = self._demonstrations_rh[0:self.task._n_steps_lookahead+1].flatten()
        # demo = np.concatenate((demo_lh, demo_rh)).flatten()
        # demo_spec = specs.Array(
        #     shape=demo.shape, dtype=demo.dtype, name='demo'
        # )
        # self._observation_spec['demo'] = demo_spec

        # self._add_end_effector_pos_mimic_reward()
        self._prior_action = None
        self._lh_target = None
        self._rh_target = None
        self._mimic_reward = 0
        self._external_demo = external_demo
        self.current_demo_lh = None
        self.current_demo_rh = None

    def observation_spec(self):
        # print("observation_spec")
        return self._observation_spec
    
    def _add_prior_action_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        print("add_prior_action_observation")
        prior_qpos = self._get_prior_action()
        self._prior_action = self.qpos2ctrl(prior_qpos)
        return timestep._replace(
            observation=collections.OrderedDict(
                timestep.observation, **{"prior_action": self._prior_action}
            )
        )
    
    def _add_demo_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        print("add_demo_observation")
        if self._external_demo:
            if self.current_demo_lh is not None and self.current_demo_rh is not None:
                demo_lh = self.current_demo_lh[0:self.task._n_steps_lookahead+1]
                demo_rh = self.current_demo_rh[0:self.task._n_steps_lookahead+1]
                self.current_demo_lh = None
                self.current_demo_rh = None
            else:
                raise ValueError("External demo is enabled but no demo is provided.")
        else:
            # print(self._demonstrations_lh[0])
            # print(self._demonstrations_rh[0])
            demo_lh = self._demonstrations_lh[self._reference_frame_idx:self._reference_frame_idx+self.task._n_steps_lookahead+1]
            demo_rh = self._demonstrations_rh[self._reference_frame_idx:self._reference_frame_idx+self.task._n_steps_lookahead+1]
            if self._reference_frame_idx + self.task._n_steps_lookahead >= self._demonstrations_length:
                # Fill rest with the last frame
                demo_lh = np.concatenate((demo_lh, self._demonstrations_lh[-1].reshape(1, 3, 6).repeat(self._reference_frame_idx + self.task._n_steps_lookahead - self._demonstrations_length + 1, axis=0)))
                demo_rh = np.concatenate((demo_rh, self._demonstrations_rh[-1].reshape(1, 3, 6).repeat(self._reference_frame_idx + self.task._n_steps_lookahead - self._demonstrations_length + 1, axis=0)))
        demo_lh = np.transpose(demo_lh, (0, 2, 1)).flatten()
        demo_rh = np.transpose(demo_rh, (0, 2, 1)).flatten()
        demo = np.concatenate((demo_lh, demo_rh)).flatten()
        return timestep._replace(
            observation=collections.OrderedDict(
                timestep.observation, **{"demo": demo}
            )
        )
        # return timestep

    def set_current_demo(self, demonstrations_lh, demonstrations_rh):
        print("set_current_demo")
        self.current_demo_lh = demonstrations_lh # (seq_len, 3, 6)
        self.current_demo_rh = demonstrations_rh # (seq_len, 3, 6)

    def _get_prior_action(self) -> np.ndarray:
        print("_get_prior_action")
        if self._external_demo:
            # print("inner:", self._demonstrations_lh[max(0, self._reference_frame_idx)])
            # print("outer:", self.current_demo_lh[0])
            # raise ValueError("External demo is enabled but no demo is provided.")
            if self.current_demo_lh is not None and self.current_demo_rh is not None:
                qvel_left, lh_dof_indices, self._lh_target = move_fingers_to_pos_qp(self,
                                            self.current_demo_lh[0], # (3, 6)
                                            finger_names=['th', 'ff', 'mf', 'rf', 'lf'],
                                            hand_side='left',
                                            targeting_wrist=False,
                                            )
                
                qvel_right, rh_dof_indices, self._rh_target = move_fingers_to_pos_qp(self,
                                            self.current_demo_rh[0], # (3, 6)
                                            finger_names=['th', 'ff', 'mf', 'rf', 'lf'],
                                            hand_side='right',
                                            targeting_wrist=False,
                                            )
            else:
                raise ValueError("External demo is enabled but no demo is provided.")
        else:
            # print(self._demonstrations_lh.shape) #(600, 3, 6)
            # print(self._demonstrations_lh[max(0, self._reference_frame_idx)].shape) #(3, 6)
            # print("lh")
            # print(max(0, self._reference_frame_idx))
            if max(0, self._reference_frame_idx) < 5:
                print(max(0, self._reference_frame_idx))
                print(self._demonstrations_lh[max(0, self._reference_frame_idx)])
            qvel_left, lh_dof_indices, self._lh_target = move_fingers_to_pos_qp(self,
                                        self._demonstrations_lh[max(0, self._reference_frame_idx)],
                                        finger_names=['th', 'ff', 'mf', 'rf', 'lf'],
                                        hand_side='left',
                                        targeting_wrist=False,
                                        )
            if max(0, self._reference_frame_idx) < 5:
                print(qvel_left)
                print(qvel_left.shape)
            qvel_right, rh_dof_indices, self._rh_target = move_fingers_to_pos_qp(self,
                                        self._demonstrations_rh[max(0, self._reference_frame_idx)],
                                        finger_names=['th', 'ff', 'mf', 'rf', 'lf'],
                                        hand_side='right',
                                        targeting_wrist=False,
                                        )
        v_full = np.zeros(self.physics.model.nv, dtype=self.physics.data.qpos.dtype)
        v_full[lh_dof_indices] = qvel_left
        v_full[rh_dof_indices] = qvel_right
        pos = self.physics.data.qpos.copy()
        mjlib.mj_integratePos(self.physics.model.ptr, pos, v_full, 0.05)
        # return pos[88:]
        return pos

    def qpos2ctrl(self, qpos):
        print("qpos2ctrl")
        # right hand
        action_ik = np.zeros(38, dtype=np.float64)
        action_ik[0:2] = qpos[90:92] # WRJ2, WRJ1
        action_ik[2:5] = qpos[108:111] # THJ4, THJ3, THJ2
        action_ik[5:7] = qpos[92:94] # FFJ4, FFJ3
        action_ik[7] = qpos[94] + qpos[95] # FFJ0
        action_ik[8:10] = qpos[96:98] # MFJ4, MFJ3
        action_ik[10] = qpos[98] + qpos[99] # MFJ0
        action_ik[11:13] = qpos[100:102] # RFJ4, RFJ3
        action_ik[13] = qpos[102] + qpos[103] # RFJ0
        action_ik[14:16] = qpos[104:106] # LFJ4, LFJ3
        action_ik[16] = qpos[106] + qpos[107] # LFJ0
        action_ik[17:19] = qpos[88:90] # forearm_tx, forearm_ty
        # left hand
        action_ik[19:21] = qpos[113:115] # WRJ2, WRJ1
        action_ik[21:24] = qpos[131:134] # THJ4, THJ3, THJ2
        action_ik[24:26] = qpos[115:117] # FFJ4, FFJ3
        action_ik[26] = qpos[117] + qpos[118] # FFJ0
        action_ik[27:29] = qpos[119:121] # MFJ4, MFJ3
        action_ik[29] = qpos[121] + qpos[122] # MFJ0
        action_ik[30:32] = qpos[123:125] # RFJ4, RFJ3
        action_ik[32] = qpos[125] + qpos[126] # RFJ0
        action_ik[33:35] = qpos[127:129] # LFJ4, LFJ3
        action_ik[35] = qpos[129] + qpos[130] # LFJ0
        action_ik[36:38] = qpos[111:113] # forearm_tx, forearm_ty

        return action_ik
    
    def step(self, action) -> dm_env.TimeStep:
        print("step")
        if self._enable_ik:
            # action_hand = action[:-1] + self._prior_action # Apply residual
            action_hand = self._prior_action
        else:
            print("ik is disabled")
            action_hand = action[:-1]
        self.non_residual_action = action_hand
        # self.physics.data.qpos[88:] = action_hand # Apply qpos instead of ctrl. Only sustain pedal is acted in task.before_step().
        action_sustain = action[-1]
        # Merge action_sustain into action_hand 
        action = np.append(action_hand, action_sustain)
        timestep = self._environment.step(action)
        self._reference_frame_idx = int(min(self._reference_frame_idx+self._step_scale, self._demonstrations_length-1))
        return self._add_demo_observation(self._add_prior_action_observation(timestep))
    
    def get_non_residual_action(self):
        print("get_non_residual_action")
        return self.non_residual_action

    def reset(self) -> dm_env.TimeStep:
        print("reset")
        timestep = self._environment.reset()
        self._mimic_reward = 0
        if self._rsi:
            self._reference_frame_idx = random.randint(-int(round(self._environment.task._initial_buffer_time/
                                               self._environment.task.control_timestep)), self._demonstrations_length-1)
            self._environment._reference_frame_idx = self._reference_frame_idx
            self._environment.task._t_idx = int(self._reference_frame_idx/self._step_scale)
            reference_joint_pos = self._get_prior_action()
            action = self.qpos2ctrl(reference_joint_pos)
            self._environment.task.right_hand.configure_joints(self.physics, action[:27])
            self._environment.task.left_hand.configure_joints(self.physics, action[27:])
        else: 
            self._reference_frame_idx = -int(round(self._environment.task._initial_buffer_time/
                                               self._environment.task.control_timestep))
        return self._add_demo_observation(self._add_prior_action_observation(timestep))

    
